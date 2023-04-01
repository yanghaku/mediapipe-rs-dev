#[cfg(feature = "audio")]
pub mod audio;

#[cfg(feature = "text")]
pub mod text;

#[cfg(feature = "vision")]
pub mod vision;

use crate::model::ModelResourceTrait;
use crate::Error;

/// Every media such as Image, Audio, Text, can implement this trait and be used as model input
pub trait ToTensor {
    fn to_tensors(
        &self,
        input_index: usize,
        model_resource: &Box<dyn ModelResourceTrait>,
        output_buffers: &mut [impl AsMut<[u8]>],
    ) -> Result<(), Error>;
}

/// Used for stream data, such video, audio.
pub trait ToTensorStream<'a> {
    type Iter: ToTensorStreamIterator;

    fn to_tensors_stream(
        &'a self,
        input_index: usize,
        model_resource: &'a Box<dyn ModelResourceTrait>,
    ) -> Result<Self::Iter, Error>;
}

/// Used for [`ToTensorStream`], a iter for to_tensors, return the timestamp_ms
pub trait ToTensorStreamIterator {
    fn next_tensors(&mut self, output_buffers: &mut [impl AsMut<[u8]>]) -> Option<u64>;

    // todo: async api
}

pub(crate) enum ToTensorInfo<'buf> {
    #[cfg(feature = "audio")]
    Audio(audio::AudioToTensorInfo),
    #[cfg(feature = "vision")]
    Image(vision::ImageToTensorInfo),
    #[cfg(feature = "text")]
    Text(text::TextToTensorInfo<'buf>),

    #[cfg(not(feature = "text"))]
    _None(std::marker::PhantomData<&'buf ()>),
}
