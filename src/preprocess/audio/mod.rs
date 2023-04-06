use super::*;
use crate::TensorType;
// audio to tensor impl
mod audio_data_to_tensor;

mod audio_raw_data;
pub use audio_raw_data::AudioRawData;

mod symphonia_data;
pub use symphonia_data::SymphoniaAudioData;

#[cfg(feature = "ffmpeg")]
mod ffmpeg_data;
#[cfg(feature = "ffmpeg")]
pub use ffmpeg_data::FFMpegAudioData;

/// Necessary information for the audio to tensor.
#[derive(Debug)]
pub struct AudioToTensorInfo {
    /// Expected audio dimensions.
    /// Expected number of channels of the input audio buffer, e.g., num_channels=1,
    pub num_channels: usize,

    ///  Expected number of samples per channel of the input audio buffer, e.g., num_samples=15600.
    pub num_samples: usize,

    /// Expected sample rate, e.g., sample_rate=16000 for 16kHz.
    pub sample_rate: usize,

    /// The number of the overlapping samples per channel between adjacent input tensors.
    pub num_overlapping_samples: usize,

    /// Expected input tensor type, e.g., tensor_type=TensorType_FLOAT32.
    pub tensor_type: TensorType,
}

/// Every Audio input impl the [`AudioData`] can use [`AudioDataToTensorIter`] to convert to tensor.
pub trait AudioData {
    /// return (sample_rate, num_samples), save the sample in sample_buffer,
    /// sample data must be range in ```[-1.0,1.0]```.
    fn next_package(
        &mut self,
        sample_buffer: &mut Vec<Vec<f32>>,
    ) -> Result<Option<(usize, usize)>, Error>;
}

impl<'model, T> InToTensorsIterator<'model> for T
where
    T: AudioData + 'model,
{
    type Iter = AudioDataToTensorIter<'model, T>;

    #[inline(always)]
    fn into_tensors_iter(self, to_tensor_info: &'model ToTensorInfo) -> Result<Self::Iter, Error> {
        let audio_to_tensor_info = to_tensor_info.try_to_audio()?;
        AudioDataToTensorIter::new(audio_to_tensor_info, self)
    }
}

/// Used for Audio To Tensor, such as [`AudioRawData`], [`SymphoniaAudioData`], [`FFMpegAudioData`], etc.
pub struct AudioDataToTensorIter<'a, Source: AudioData = SymphoniaAudioData> {
    audio_to_tensor_info: &'a AudioToTensorInfo,
    source: Source,
    input_buffer: Vec<Vec<f32>>,
    process_buffer: Vec<std::collections::VecDeque<f32>>,
    input_num_channels: usize,
    input_sample_rate: usize,
    processed_time_stamp_ms: u64,
}
