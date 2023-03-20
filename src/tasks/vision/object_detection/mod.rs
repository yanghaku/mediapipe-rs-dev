mod builder;
pub use builder::ObjectDetectorBuilder;

use crate::model_resource::ModelResourceTrait;
use crate::postprocess::sessions::DetectionSession;
use crate::postprocess::DetectionResult;
use crate::preprocess::ToTensor;
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

        let detection_session = DetectionSession::new(
            &self.build_info.classifier_builder,
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
        })
    }

    #[inline(always)]
    pub fn classify<'t>(self, input: &impl ToTensor<'t>) -> Result<DetectionResult, Error> {
        self.new_session()?.detect(input)
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
}

impl<'a> ObjectDetectorSession<'a> {
    pub fn detect<'t>(&mut self, input: &impl ToTensor<'t>) -> Result<DetectionResult, Error> {
        let tensor = input.to_tensor(0, &self.detector.model_resource)?;

        self.execution_ctx.set_input(
            0,
            self.detector.input_tensor_type,
            self.input_tensor_shape,
            tensor.as_ref(),
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
}
