use image::{ImageBuffer, Luma};

/// The output result of Image Segmentation tasks.
#[derive(Debug)]
pub struct ImageSegmenterResult {
    /// Multiple masks of float image in VEC32F1 format where, for each mask, each
    /// pixel represents the prediction confidence, usually in the [0, 1] range.
    pub confidence_masks: Option<Vec<ImageBuffer<Luma<f32>, Vec<f32>>>>,

    /// A category mask of uint8 image in GRAY8 format where each pixel represents
    /// the class which the pixel in the original image was predicted to belong to.
    pub category_mask: Option<ImageBuffer<Luma<u8>, Vec<u8>>>,
}
