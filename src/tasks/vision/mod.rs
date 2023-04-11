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
        video_data: &mut impl crate::preprocess::vision::VideoData,
    ) -> Result<Option<Self::Result>, crate::Error>;
}
