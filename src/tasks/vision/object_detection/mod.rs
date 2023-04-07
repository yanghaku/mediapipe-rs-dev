mod builder;
pub use builder::ObjectDetectorBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::sessions::{CategoriesFilter, DetectionSession};
use crate::postprocess::{DetectionResult, ResultsIter};
use crate::preprocess::{InToTensorsIterator, Tensor, TensorsIterator};
use crate::tasks::TaskSession;
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs object detection on single images, video frames, or live stream.
pub struct ObjectDetector {
    build_info: ObjectDetectorBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    bound_box_properties: [usize; 4],
    location_buf_index: usize,
    categories_buf_index: usize,
    score_buf_index: usize,
    num_box_buf_index: usize,
    // only one input and one output
    input_tensor_type: TensorType,
}

macro_rules! get_type_and_quantization {
    ( $self:ident, $index:expr ) => {{
        let t =
            model_resource_check_and_get_impl!($self.model_resource, output_tensor_type, $index);
        let q = $self
            .model_resource
            .output_tensor_quantization_parameters($index);
        check_quantization_parameters!(t, q, $index);

        (t, q)
    }};
}

impl ObjectDetector {
    base_task_build_info_get_impl!();

    classifier_build_info_get_impl!();

    #[inline(always)]
    pub fn new_session(&self) -> Result<ObjectDetectorSession, Error> {
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);
        let labels = self.model_resource.output_tensor_labels_locale(
            self.categories_buf_index,
            self.build_info
                .classifier_builder
                .display_names_locale
                .as_ref(),
        )?;

        let categories_filter =
            CategoriesFilter::new(&self.build_info.classifier_builder, labels.0, labels.1);
        let detection_session = DetectionSession::new(
            categories_filter,
            self.build_info.classifier_builder.max_results,
            &self.bound_box_properties,
            get_type_and_quantization!(self, self.location_buf_index),
            get_type_and_quantization!(self, self.categories_buf_index),
            get_type_and_quantization!(self, self.score_buf_index),
        );

        let execution_ctx = self.graph.init_execution_context()?;
        Ok(ObjectDetectorSession {
            detector: self,
            execution_ctx,
            detection_session,
            num_box_buf: [0f32],
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }

    /// Detect one image.
    #[inline(always)]
    pub fn detect(&self, input: &impl Tensor) -> Result<DetectionResult, Error> {
        self.new_session()?.detect(input)
    }

    /// Detect input video stream, and collect all results to [`Vec`]
    #[inline(always)]
    pub fn detect_for_video<'a>(
        &'a self,
        input_stream: impl InToTensorsIterator<'a>,
    ) -> Result<Vec<DetectionResult>, Error> {
        let iter = self.detection_results_iter(input_stream)?;
        let mut session = self.new_session()?;
        iter.to_vec(&mut session)
    }

    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn detection_results_iter<'a, T>(
        &'a self,
        input_stream: T,
    ) -> Result<ResultsIter<ObjectDetectorSession<'_>, T::Iter>, Error>
    where
        T: InToTensorsIterator<'a>,
    {
        let to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0);
        let input_tensors_iter = input_stream.into_tensors_iter(to_tensor_info)?;
        Ok(ResultsIter::new(input_tensors_iter))
    }
}

/// Session to run inference. If process multiple images, use it can get better performance.
///
/// ```rust
/// use mediapipe_rs::tasks::vision::ObjectDetector;
///
/// let object_detector: ObjectDetector;
/// let mut session = object_detector.new_session()?;
/// for image in images {
///     session.detect(image)?;
/// }
/// ```
pub struct ObjectDetectorSession<'a> {
    detector: &'a ObjectDetector,
    execution_ctx: GraphExecutionContext<'a>,
    detection_session: DetectionSession<'a>,

    num_box_buf: [f32; 1],
    input_tensor_shape: &'a [usize],
    input_buffer: Vec<u8>,
}

impl<'a> ObjectDetectorSession<'a> {
    // todo: usage the timestamp
    #[allow(unused)]
    #[inline(always)]
    fn compute(&mut self, timestamp_ms: Option<u64>) -> Result<DetectionResult, Error> {
        self.execution_ctx.set_input(
            0,
            self.detector.input_tensor_type,
            self.input_tensor_shape,
            self.input_buffer.as_ref(),
        )?;
        self.execution_ctx.compute()?;

        // get num box
        let output_size = self
            .execution_ctx
            .get_output(self.detector.num_box_buf_index, &mut self.num_box_buf)?;
        if output_size != 4 {
            return Err(Error::ModelInconsistentError(format!(
                "Model output bytes size is `{}`, but got `{}`",
                4, output_size
            )));
        }
        let num_box = self.num_box_buf[0].round() as usize;

        // realloc
        self.detection_session.realloc(num_box, 0);

        // get other buffers
        self.execution_ctx.get_output(
            self.detector.location_buf_index,
            self.detection_session.location_buf(),
        )?;
        self.execution_ctx.get_output(
            self.detector.categories_buf_index,
            self.detection_session.categories_buf(),
        )?;
        self.execution_ctx.get_output(
            self.detector.score_buf_index,
            self.detection_session.score_buf(),
        )?;

        // generate result
        Ok(self.detection_session.result(num_box, 0))
    }

    /// Detect one image
    #[inline(always)]
    pub fn detect(&mut self, input: &impl Tensor) -> Result<DetectionResult, Error> {
        let to_tensor_info =
            model_resource_check_and_get_impl!(self.detector.model_resource, to_tensor_info, 0);
        input.to_tensors(to_tensor_info, &mut [&mut self.input_buffer])?;
        self.compute(None)
    }

    /// Detect input video stream use this session, and collect all results to [`Vec`]
    #[inline(always)]
    pub fn detect_for_video(
        &'a mut self,
        input_stream: impl InToTensorsIterator<'a>,
    ) -> Result<Vec<DetectionResult>, Error> {
        self.detector
            .detection_results_iter(input_stream)?
            .to_vec(self)
    }
}

impl<'a> TaskSession for ObjectDetectorSession<'a> {
    type Result = DetectionResult;

    #[inline]
    fn process_next<TensorsIter: TensorsIterator>(
        &mut self,
        input_tensors_iter: &mut TensorsIter,
    ) -> Result<Option<Self::Result>, Error> {
        if let Some(timestamp_ms) =
            input_tensors_iter.next_tensors(&mut [&mut self.input_buffer])?
        {
            return Ok(Some(self.compute(Some(timestamp_ms))?));
        }
        Ok(None)
    }
}
