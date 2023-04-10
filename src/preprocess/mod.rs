mod common;

#[cfg(feature = "audio")]
pub mod audio;

#[cfg(feature = "text")]
pub mod text;

#[cfg(feature = "vision")]
pub mod vision;

use crate::Error;

/// Every media such as Image, Text, can implement this trait and be used as model input
pub trait Tensor {
    fn to_tensors(
        &self,
        to_tensor_info: &ToTensorInfo,
        output_buffers: &mut [impl AsMut<[u8]>],
    ) -> Result<(), Error>;
}

/// Every media such as Video, Audio, can implement this trait and be used as model input
pub trait InToTensorsIterator<'tensor> {
    type Iter: TensorsIterator + 'tensor;

    fn into_tensors_iter<'model: 'tensor>(
        self,
        to_tensor_info: &'model ToTensorInfo,
    ) -> Result<Self::Iter, Error>;
}

/// Used for stream data, such video, audio.
pub trait TensorsIterator {
    /// get next tensors save to output_buffers, return timestamp_ms
    /// if the stream is end, return None
    fn next_tensors(
        &mut self,
        output_buffers: &mut [impl AsMut<[u8]>],
    ) -> Result<Option<u64>, Error>;

    // todo: async api
}

#[derive(Debug)]
enum ToTensorInfoInner<'buf> {
    #[cfg(feature = "audio")]
    Audio(audio::AudioToTensorInfo),
    #[cfg(feature = "vision")]
    Image(vision::ImageToTensorInfo),
    #[cfg(feature = "text")]
    Text(text::TextToTensorInfo<'buf>),

    None(#[cfg(not(feature = "text"))] std::marker::PhantomData<&'buf ()>),
}

#[derive(Debug)]
pub struct ToTensorInfo<'buf> {
    inner: ToTensorInfoInner<'buf>,
}

impl<'buf> ToTensorInfo<'buf> {
    #[inline(always)]
    pub fn new_none() -> Self {
        #[cfg(not(feature = "text"))]
        return Self {
            inner: ToTensorInfoInner::None(Default::default()),
        };
        #[cfg(feature = "text")]
        return Self {
            inner: ToTensorInfoInner::None(),
        };
    }

    #[cfg(feature = "audio")]
    #[inline(always)]
    pub fn new_audio(audio_to_tensor_info: audio::AudioToTensorInfo) -> Self {
        Self {
            inner: ToTensorInfoInner::Audio(audio_to_tensor_info),
        }
    }

    #[cfg(feature = "vision")]
    #[inline(always)]
    pub fn new_image(image_to_tensor_info: vision::ImageToTensorInfo) -> Self {
        Self {
            inner: ToTensorInfoInner::Image(image_to_tensor_info),
        }
    }

    #[cfg(feature = "text")]
    #[inline(always)]
    pub fn new_text(text_to_tensor_info: text::TextToTensorInfo<'buf>) -> Self {
        Self {
            inner: ToTensorInfoInner::Text(text_to_tensor_info),
        }
    }

    #[cfg(feature = "audio")]
    #[inline(always)]
    pub fn try_to_audio(&self) -> Result<&audio::AudioToTensorInfo, Error> {
        match &self.inner {
            ToTensorInfoInner::Audio(a) => Ok(a),
            _ => {
                return Err(Error::ModelInconsistentError(format!(
                    "Expect Audio to Tensor Info, but got `{:?}`",
                    self.inner
                )));
            }
        }
    }

    #[cfg(feature = "vision")]
    #[inline(always)]
    pub fn try_to_image(&self) -> Result<&vision::ImageToTensorInfo, Error> {
        match &self.inner {
            ToTensorInfoInner::Image(i) => Ok(i),
            _ => {
                return Err(Error::ModelInconsistentError(format!(
                    "Expect Image to Tensor Info, but got `{:?}`",
                    self.inner
                )));
            }
        }
    }

    #[cfg(feature = "text")]
    #[inline(always)]
    pub fn try_to_text(&self) -> Result<&text::TextToTensorInfo<'buf>, Error> {
        match &self.inner {
            ToTensorInfoInner::Text(t) => Ok(t),
            _ => {
                return Err(Error::ModelInconsistentError(format!(
                    "Expect Text to Tensor Info, but got `{:?}`",
                    self.inner
                )));
            }
        }
    }
}
