mod builder;
pub use builder::HandDetectorBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{
    Anchor, DetectionBoxFormat, DetectionResult, NonMaxSuppressionAlgorithm,
    NonMaxSuppressionOverlapType, TensorsToDetection,
};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

pub struct HandDetector {
    build_options: HandDetectorBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    anchors: Vec<Anchor>,
    location_buf_index: usize,
    score_buf_index: usize,
    num_box: usize,

    // only one input and one output
    input_tensor_type: TensorType,
}

impl HandDetector {
    detector_impl!(HandDetectorSession, DetectionResult);

    #[inline(always)]
    pub fn num_hands(&self) -> i32 {
        self.build_options.num_hands
    }

    #[inline(always)]
    pub fn min_detection_confidence(&self) -> f32 {
        self.build_options.min_detection_confidence
    }

    #[inline(always)]
    pub fn new_session(&self) -> Result<HandDetectorSession, Error> {
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);

        let mut tensors_to_detection = TensorsToDetection::new_with_anchors(
            &self.anchors,
            self.min_detection_confidence(),
            self.num_hands(),
            get_type_and_quantization!(self, self.location_buf_index),
            get_type_and_quantization!(self, self.score_buf_index),
        );

        // config options
        tensors_to_detection.set_anchors_scales(192.0, 192.0, 192.0, 192.0);
        tensors_to_detection.set_num_coords(18);
        tensors_to_detection.set_key_points(7, 2, 4);
        tensors_to_detection.set_sigmoid_score(true);
        tensors_to_detection.set_score_clipping_thresh(100.);
        tensors_to_detection.set_box_format(DetectionBoxFormat::XYWH);
        tensors_to_detection.set_nms_min_suppression_threshold(0.3);
        tensors_to_detection
            .set_nms_overlap_type(NonMaxSuppressionOverlapType::IntersectionOverUnion);
        tensors_to_detection.set_nms_algorithm(NonMaxSuppressionAlgorithm::WEIGHTED);
        tensors_to_detection.realloc(self.num_box);

        let execution_ctx = self.graph.init_execution_context()?;
        Ok(HandDetectorSession {
            detector: self,
            execution_ctx,
            tensors_to_detection,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }
}

pub struct HandDetectorSession<'model> {
    detector: &'model HandDetector,
    execution_ctx: GraphExecutionContext<'model>,
    tensors_to_detection: TensorsToDetection<'model>,

    input_tensor_shape: &'model [usize],
    input_buffer: Vec<u8>,
}

impl<'model> HandDetectorSession<'model> {
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

        self.execution_ctx.get_output(
            self.detector.location_buf_index,
            self.tensors_to_detection.location_buf(),
        )?;
        self.execution_ctx.get_output(
            self.detector.score_buf_index,
            self.tensors_to_detection.score_buf(),
        )?;

        // generate result
        Ok(self.tensors_to_detection.result(self.detector.num_box))
    }

    detector_session_impl!(DetectionResult);
}

detection_task_session_impl!(HandDetectorSession, DetectionResult);
