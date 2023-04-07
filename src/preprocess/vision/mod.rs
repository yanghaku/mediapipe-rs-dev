mod image;

#[cfg(feature = "ffmpeg")]
mod ffmpeg;
#[cfg(feature = "ffmpeg")]
pub use ffmpeg::FFMpegVideoData;

use super::*;
use crate::TensorType;

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
