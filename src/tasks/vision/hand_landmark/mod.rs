mod builder;

use super::{HandDetector, HandDetectorBuilder, HandDetectorSession};
pub use builder::HandLandmarkerBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{HandLandmarkResult, ResultsIter};
use crate::preprocess::{InToTensorsIterator, Tensor, TensorsIterator};
use crate::tasks::TaskSession;
use crate::{Error, Graph, GraphExecutionContext, TensorType};

pub struct HandLandmarker {
    pub(super) build_options: HandLandmarkerBuilder,
    pub(super) model_resource: Box<dyn ModelResourceTrait>,
    pub(super) graph: Graph,

    pub(super) subtask: HandDetector,

    pub(super) handedness_buf_index: usize,
    pub(super) score_buf_index: usize,
    pub(super) landmarks_buf_index: usize,
    pub(super) world_landmarks_buf_index: usize,

    // only one input and one output
    pub(super) input_tensor_type: TensorType,
}

impl HandLandmarker {
    base_task_options_get_impl!();

    hand_landmark_options_get_impl!();

    #[inline(always)]
    pub fn new_session(&self) -> Result<HandLandmarkerSession, Error> {
        todo!()
    }

    /// Detect one image.
    #[inline(always)]
    pub fn detect(&self, input: &impl Tensor) -> Result<HandLandmarkResult, Error> {
        todo!()
        // self.new_session()?.detect(input)
    }

    /// Detect input video stream, and collect all results to [`Vec`]
    #[inline(always)]
    pub fn detect_for_video<'a>(
        &'a self,
        input_stream: impl InToTensorsIterator<'a>,
    ) -> Result<Vec<HandLandmarkResult>, Error> {
        let iter = self.detection_results_iter(input_stream)?;
        let mut session = self.new_session()?;
        iter.to_vec(&mut session)
    }

    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn detection_results_iter<'a, T>(
        &'a self,
        input_stream: T,
    ) -> Result<ResultsIter<HandLandmarkerSession<'_>, T::Iter>, Error>
    where
        T: InToTensorsIterator<'a>,
    {
        let to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0);
        let input_tensors_iter = input_stream.into_tensors_iter(to_tensor_info)?;
        Ok(ResultsIter::new(input_tensors_iter))
    }
}

pub struct HandLandmarkerSession<'a> {
    land: &'a HandLandmarker,
}

impl<'a> TaskSession for HandLandmarkerSession<'a> {
    type Result = HandLandmarkResult;

    fn process_next<TensorsIter: TensorsIterator>(
        &mut self,
        input_tensors_iter: &mut TensorsIter,
    ) -> Result<Option<Self::Result>, Error> {
        todo!()
    }
}
