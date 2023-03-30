use super::*;
use crate::{ModelResourceTrait, TensorType};

/// Necessary information for the audio to tensor.
#[derive(Debug)]
pub struct AudioToTensorInfo {
    /// Expected audio dimensions.
    /// Expected number of channels of the input audio buffer, e.g., num_channels=1,
    pub num_channels: u32,

    ///  Expected number of samples per channel of the input audio buffer, e.g., num_samples=15600.
    pub num_samples: u32,

    /// Expected sample rate, e.g., sample_rate=16000 for 16kHz.
    pub sample_rate: u32,

    /// The number of the overlapping samples per channel between adjacent input tensors.
    pub num_overlapping_samples: u32,

    /// Expected input tensor type, e.g., tensor_type=TensorType_FLOAT32.
    pub tensor_type: TensorType,
}

pub struct AudioData<T = Vec<Vec<f32>>, E = Vec<f32>>
where
    T: AsRef<[E]>,
    E: AsRef<[f32]>,
{
    sample_rate: u32,
    buf: T,
    _marker: std::marker::PhantomData<E>,
}

impl<T, E> AudioData<T, E>
where
    T: AsRef<[E]>,
    E: AsRef<[f32]>,
{
    pub fn new(raw_major_matrix: T, sample_rate: u32) -> Self {
        Self {
            buf: raw_major_matrix,
            sample_rate,
            _marker: Default::default(),
        }
    }

    #[inline(always)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline(always)]
    pub fn buf(&self) -> &T {
        &self.buf
    }

    #[inline(always)]
    pub fn num_channels(&self) -> usize {
        self.buf.as_ref().len()
    }
}

impl<'a, T, E> ToTensorStream<'a> for AudioData<T, E>
where
    T: AsRef<[E]>,
    E: AsRef<[f32]>,
{
    type Iter = AudioDataToTensorIter<'a>;

    #[inline]
    fn to_tensors_stream(
        &'a self,
        input_index: usize,
        model_resource: &'a Box<dyn ModelResourceTrait>,
    ) -> Result<Self::Iter, Error> {
        AudioDataToTensorIter::new(self, input_index, model_resource)
    }
}

enum BufferData<'a> {
    Owned(Vec<Vec<f32>>),
    Borrowed(Vec<&'a [f32]>),
}

#[doc(hidden)]
pub struct AudioDataToTensorIter<'a> {
    audio_to_tensor_info: &'a AudioToTensorInfo,
    matrix: BufferData<'a>,
    now_index: usize,
    timestamp_interval_ms: u32,
}

impl<'a> ToTensorStreamIterator for AudioDataToTensorIter<'a> {
    fn next_tensors(&mut self, output_buffers: &mut [impl AsMut<[u8]>]) -> Option<u64> {
        debug_assert_eq!(output_buffers.len(), 0);
        let output_buffer = &mut output_buffers[0];

        // todo: check if need fft for model
        let index = self.now_index * self.audio_to_tensor_info.num_samples as usize;
        let index_end = index + self.audio_to_tensor_info.num_samples as usize;
        let data: Vec<&[f32]> = match &self.matrix {
            BufferData::Owned(v) => {
                if index >= v[0].len() {
                    return None;
                }
                v.iter()
                    .map(|c| &c[index..std::cmp::min(index_end, c.len())])
                    .collect()
            }
            BufferData::Borrowed(v) => {
                if index >= v[0].len() {
                    return None;
                }
                v.iter()
                    .map(|c| &c[index..std::cmp::min(index_end, c.len())])
                    .collect()
            }
        };
        Self::output_to_tensor(self.audio_to_tensor_info.tensor_type, data, output_buffer);

        let timestamp = self.now_index as u64 * self.timestamp_interval_ms as u64;
        self.now_index += 1;
        Some(timestamp)
    }
}

impl<'a> AudioDataToTensorIter<'a> {
    #[inline]
    fn new<T, E>(
        audio_data: &'a AudioData<T, E>,
        input_index: usize,
        model_resource: &'a Box<dyn ModelResourceTrait>,
    ) -> Result<Self, Error>
    where
        T: AsRef<[E]>,
        E: AsRef<[f32]>,
    {
        let audio_to_tensor_info =
            model_resource_check_and_get_impl!(model_resource, audio_to_tensor_info, input_index);
        match audio_to_tensor_info.tensor_type {
            // reference: https://github.com/google/mediapipe/blob/master/mediapipe/tasks/cc/audio/utils/audio_tensor_specs.cc
            TensorType::F16 | TensorType::F32 => {}
            _ => {
                return Err(Error::ModelInconsistentError(
                    "Model only support F32 or F16 input now.".into(),
                ));
            }
        };

        let channels = audio_data.num_channels();
        if channels == 0 {
            return Err(Error::ArgumentError("num channels cannot be `0`".into()));
        }
        let mono_output = audio_to_tensor_info.num_channels == 1;
        let channels_match = channels != audio_to_tensor_info.num_channels as usize;
        if !mono_output && !channels_match {
            return Err(Error::ArgumentError(format!(
                "Audio input has `{}` channel(s) but the model requires `{}` channel(s)",
                channels, audio_to_tensor_info.num_channels
            )));
        }

        let buf = if !channels_match {
            // cal the mean
            let mut temp_vec = audio_data.buf.as_ref()[0].as_ref().to_vec();
            for i in 1..channels {
                let samples = audio_data.buf.as_ref()[i].as_ref();
                for j in 0..samples.len() {
                    temp_vec[j] += samples[j];
                }
            }
            let div = channels as f32;
            temp_vec.iter_mut().for_each(|c| *c /= div);
            BufferData::Owned(vec![temp_vec])
        } else {
            let mut vec = Vec::with_capacity(channels);
            for p in audio_data.buf.as_ref() {
                vec.push(p.as_ref())
            }
            BufferData::Borrowed(vec)
        };

        if audio_data.sample_rate != audio_to_tensor_info.sample_rate {
            todo!("resample");
        }

        let timestamp_interval_ms = (audio_to_tensor_info.num_samples as f32
            / audio_to_tensor_info.sample_rate as f32
            * 1000f32) as u32;

        Ok(Self {
            audio_to_tensor_info,
            matrix: buf,
            timestamp_interval_ms,
            now_index: 0,
        })
    }

    fn output_to_tensor(
        tensor_type: TensorType,
        data: Vec<&[f32]>,
        output_buffer: &mut impl AsMut<[u8]>,
    ) {
        match tensor_type {
            TensorType::F16 => {}
            TensorType::F32 => {
                let mut index = 0;
                for d in &data {
                    let len = d.len() * std::mem::size_of::<f32>();
                    output_buffer.as_mut()[index..index + len].copy_from_slice(unsafe {
                        core::slice::from_raw_parts((*d).as_ptr() as *const u8, len)
                    });
                    index += len;
                }
                if index < output_buffer.as_mut().len() {
                    output_buffer.as_mut()[index..].fill(0);
                }
            }
            _ => {
                unreachable!()
            }
        }
    }
}
