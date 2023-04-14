use super::{HandDetectorBuilder, HandLandmarker};

use crate::model::{ModelResourceTrait, ZipFiles};
use crate::tasks::common::{BaseTaskOptions, HandLandmarkOptions};

/// Configure the properties of a new hand landmark task.
/// Methods can be chained on it in order to configure it.
pub struct HandLandmarkerBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) hand_landmark_options: HandLandmarkOptions,
}

macro_rules! check_fp32 {
    ( $model_resource:ident, $index:ident ) => {{
        let tensor_type =
            model_resource_check_and_get_impl!($model_resource, output_tensor_type, $index);
        if tensor_type != crate::TensorType::F32 {
            return Err(crate::Error::ModelInconsistentError(format!(
                "expect output `{}` type is Fp32, but got `{:?}`",
                $index, tensor_type
            )));
        }
    }};
}

impl HandLandmarkerBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            hand_landmark_options: Default::default(),
        }
    }

    base_task_options_impl!();

    hand_landmark_options_impl!();

    pub const HAND_DETECTOR_CANDIDATE_NAMES: &'static [&'static str] = &["hand_detector.tflite"];
    pub const HAND_LANDMARKS_CANDIDATE_NAMES: &'static [&'static str] =
        &["hand_landmarks_detector.tflite"];

    #[inline]
    pub fn finalize(mut self) -> Result<HandLandmarker, crate::Error> {
        hand_landmark_options_check!(self);
        let buf = base_task_options_check_and_get_buf!(self);

        let zip_file = ZipFiles::new(buf.as_ref())?;
        let landmark_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::HAND_LANDMARKS_CANDIDATE_NAMES,
            "HandLandmark"
        );
        let hand_detection_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::HAND_DETECTOR_CANDIDATE_NAMES,
            "HandDetection"
        );

        let subtask = HandDetectorBuilder::new()
            .model_asset_slice(hand_detection_file)
            .execution_target(self.base_task_options.execution_target)
            .num_hands(self.hand_landmark_options.num_hands)
            .min_detection_confidence(self.hand_landmark_options.min_hand_detection_confidence)
            .finalize()?;

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model::parse_model(landmark_file.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

        // check model
        model_base_check_impl!(model_resource, 1, 4);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        // todo: get these from metadata
        let handedness_buf_index = 2;
        let score_buf_index = 1;
        let landmarks_buf_index = 0;
        let world_landmarks_buf_index = 3;
        // now only fp32 model
        check_fp32!(model_resource, handedness_buf_index);
        check_fp32!(model_resource, score_buf_index);

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.execution_target,
        )
        .build_from_shared_slices([landmark_file])?;

        Ok(HandLandmarker {
            build_options: self,
            model_resource,
            graph,
            subtask,
            handedness_buf_index,
            score_buf_index,
            landmarks_buf_index,
            world_landmarks_buf_index,
            input_tensor_type,
        })
    }
}
