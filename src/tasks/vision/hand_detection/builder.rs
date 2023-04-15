use super::HandDetector;
use crate::model::ModelResourceTrait;
use crate::postprocess::SsdAnchorsBuilder;
use crate::tasks::common::BaseTaskOptions;

/// Configure the properties of a new object detection task.
/// Methods can be chained on it in order to configure it.
pub struct HandDetectorBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    /// The maximum number of hands output by the detector.
    pub(super) num_hands: i32,
    /// Minimum confidence value ([0.0, 1.0]) for confidence score to be considered
    /// successfully detecting a hand in the image.
    pub(super) min_detection_confidence: f32,
}

impl Default for HandDetectorBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl HandDetectorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            num_hands: 1,
            min_detection_confidence: 0.5,
        }
    }

    base_task_options_impl!();

    #[inline(always)]
    pub fn num_hands(mut self, num_hands: i32) -> Self {
        self.num_hands = num_hands;
        self
    }

    #[inline(always)]
    pub fn min_detection_confidence(mut self, min_detection_confidence: f32) -> Self {
        self.min_detection_confidence = min_detection_confidence;
        self
    }

    #[inline]
    pub fn finalize(mut self) -> Result<HandDetector, crate::Error> {
        if self.num_hands == 0 {
            return Err(crate::Error::ArgumentError(
                "The number of max hands cannot be zero".into(),
            ));
        }
        let buf = base_task_options_check_and_get_buf!(self);

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model::parse_model(buf.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

        // check model
        model_base_check_impl!(model_resource, 1, 2);
        let img_info =
            model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;

        // generate anchors
        // todo: read info from metadata
        let num_box = 2016;
        let anchors = SsdAnchorsBuilder::new(img_info.width, img_info.height, 0.1484375, 0.75, 4)
            .anchor_offset_x(0.5)
            .anchor_offset_y(0.5)
            .strides(vec![8, 16, 16, 16])
            .aspect_ratios(vec![1.0])
            .fixed_anchor_size(true)
            .generate();

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.execution_target,
        )
        .build_from_shared_slices([buf])?;

        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        return Ok(HandDetector {
            build_options: self,
            model_resource,
            graph,
            anchors,
            location_buf_index: 0,
            score_buf_index: 1,
            num_box,
            input_tensor_type,
        });
    }
}
