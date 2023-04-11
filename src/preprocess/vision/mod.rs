mod image;

#[cfg(feature = "ffmpeg")]
mod ffmpeg;
#[cfg(feature = "ffmpeg")]
pub use ffmpeg::FFMpegVideoData;

use super::*;
use crate::TensorType;

pub trait ImageToTensor {
    /// convert image to tensors, save to output_buffers
    fn to_tensor<T: AsMut<[u8]>>(
        &self,
        to_tensor_info: &ImageToTensorInfo,
        output_buffers: &mut T,
    ) -> Result<(), Error>;

    /// return image size: (weight, height)
    fn image_size(&self) -> (u32, u32);

    /// return the current timestamp (ms)
    /// video frame must return a valid timestamp
    fn time_stamp_ms(&self) -> Option<u64> {
        return None;
    }
}

/// Used for video data.
/// Now rust stable cannot use [Generic Associated Types](https://rust-lang.github.io/rfcs/1598-generic_associated_types.html)
pub trait VideoData {
    type Frame<'frame>: ImageToTensor
    where
        Self: 'frame;

    fn next_frame(&mut self) -> Result<Option<Self::Frame<'_>>, Error>;
}

/// Data layout in memory for image tensor. ```NCHW```, ```NHWC```, ```CHWN```.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum ImageDataLayout {
    NCHW,
    NHWC,
    CHWN,
}

/// Image Color Type
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum ImageColorSpaceType {
    RGB,
    GRAYSCALE,
    UNKNOWN,
}

/// Necessary information for the image to tensor.
#[derive(Debug)]
pub struct ImageToTensorInfo {
    pub image_data_layout: ImageDataLayout,
    pub color_space: ImageColorSpaceType,
    pub tensor_type: TensorType,
    pub width: u32,
    pub height: u32,
    pub stats_min: Vec<f32>,
    pub stats_max: Vec<f32>,
    /// (mean,std), len can be 1 or 3
    pub normalization_options: (Vec<f32>, Vec<f32>),
}
