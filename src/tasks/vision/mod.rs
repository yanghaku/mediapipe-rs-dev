mod hand_detection;
mod hand_landmark;
mod image_classification;
mod object_detection;

pub use hand_detection::{HandDetector, HandDetectorBuilder, HandDetectorSession};
pub use hand_landmark::{HandLandmarker, HandLandmarkerBuilder, HandLandmarkerSession};
pub use image_classification::{ImageClassifier, ImageClassifierBuilder, ImageClassifierSession};
pub use object_detection::{ObjectDetector, ObjectDetectorBuilder, ObjectDetectorSession};

/// Task session trait to process the video stream data
pub trait TaskSession {
    type Result: 'static;

    /// process the next tensors from input stream
    fn process_next(
        &mut self,
        process_options: &ImageProcessingOptions,
        video_data: &mut impl crate::preprocess::vision::VideoData,
    ) -> Result<Option<Self::Result>, crate::Error>;
}

/// Options for image processing.
///
/// If both region-or-interest and rotation are specified, the crop around the
/// region-of-interest is extracted first, then the specified rotation is applied to the crop.
#[derive(Clone, Debug)]
pub struct ImageProcessingOptions {
    pub(crate) region_of_interest: Option<crate::postprocess::Rect<f32>>,
    /// 0, 90, 180, 270
    pub(crate) rotation_degrees: i32,
}

impl Default for ImageProcessingOptions {
    #[inline(always)]
    fn default() -> Self {
        Self {
            region_of_interest: None,
            rotation_degrees: 0,
        }
    }
}

impl ImageProcessingOptions {
    /// Create default options
    #[inline(always)]
    pub fn new() -> Self {
        Default::default()
    }

    /// The optional region-of-interest to crop from the image.
    /// If not specified, the full image is used.
    ///
    /// Coordinates must be in [0,1] with 'left' < 'right' and 'top' < bottom.
    #[inline(always)]
    pub fn rotation_degrees(mut self, mut rotation_degrees: i32) -> Result<Self, crate::Error> {
        if rotation_degrees % 90 != 0 {
            return Err(crate::Error::ArgumentError(format!(
                "The rotation must be a multiple (positive or negative) of 90°, but got `{}`",
                rotation_degrees
            )));
        }
        rotation_degrees = rotation_degrees % 360;
        if rotation_degrees < 0 {
            rotation_degrees += 360;
        }
        self.rotation_degrees = rotation_degrees;
        Ok(self)
    }

    /// The rotation to apply to the image (or cropped region-of-interest), in degrees clockwise.
    ///
    /// The rotation must be a multiple (positive or negative) of 90°.
    /// default is 0.
    #[inline(always)]
    pub fn region_of_interest(
        mut self,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    ) -> Result<Self, crate::Error> {
        if top < 0. || top > 1. {
            return Err(crate::Error::ArgumentError(format!(
                "Rect top must in range [0, 1], but got `{}`",
                top
            )));
        }
        if bottom < 0. || bottom > 1. {
            return Err(crate::Error::ArgumentError(format!(
                "Rect bottom must in range [0, 1], but got `{}`",
                bottom
            )));
        }
        if left < 0. || left > 1. {
            return Err(crate::Error::ArgumentError(format!(
                "Rect left must in range [0, 1], but got `{}`",
                left
            )));
        }
        if right < 0. || right > 1. {
            return Err(crate::Error::ArgumentError(format!(
                "Rect right must in range [0, 1], but got `{}`",
                right
            )));
        }
        if left >= right {
            return Err(crate::Error::ArgumentError(format!(
                "Rect left must less than right, but got `left({})` >= `right({})`",
                left, right
            )));
        }
        if top >= bottom {
            return Err(crate::Error::ArgumentError(format!(
                "Rect top must less than bottom, but got `top({})` >= `bottom({})`",
                top, bottom
            )));
        }
        self.region_of_interest = Some(crate::postprocess::Rect {
            left,
            top,
            right,
            bottom,
        });
        Ok(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_image_process_options_check() {
        let default: ImageProcessingOptions = Default::default();
        assert_eq!(default.rotation_degrees, 0);
        assert_eq!(default.region_of_interest, None);

        assert!(ImageProcessingOptions::new().rotation_degrees(10).is_err());
        assert!(ImageProcessingOptions::new().rotation_degrees(-10).is_err());
        assert!(ImageProcessingOptions::new().rotation_degrees(-180).is_ok());
        assert!(ImageProcessingOptions::new().rotation_degrees(270).is_ok());

        assert!(ImageProcessingOptions::new()
            .region_of_interest(0.1, 0.2, 0.5, 0.7,)
            .is_ok());
        assert!(ImageProcessingOptions::new()
            .region_of_interest(-1., 1., 1., 1.,)
            .is_err());
        assert!(ImageProcessingOptions::new()
            .region_of_interest(1.1, 1., 1., 1.,)
            .is_err());
        assert!(ImageProcessingOptions::new()
            .region_of_interest(0.5, 0.4, 1., 0.3,)
            .is_err());
        assert!(ImageProcessingOptions::new()
            .region_of_interest(0.9, 0.4, 0.4, 1.,)
            .is_err());
    }
}
