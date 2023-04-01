use super::ObjectDetector;
use crate::model::ModelResourceTrait;
use crate::tasks::common::{BaseTaskBuilder, ClassifierBuilder};

/// Configure the properties of a new object detection task.
/// Methods can be chained on it in order to configure it.
pub struct ObjectDetectorBuilder {
    pub(super) base_task_builder: BaseTaskBuilder,
    pub(super) classifier_builder: ClassifierBuilder,
}

impl ObjectDetectorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_builder: Default::default(),
            classifier_builder: Default::default(),
        }
    }

    base_builder_impl!();

    classifier_builder_impl!();

    #[inline]
    pub fn finalize(mut self) -> Result<ObjectDetector, crate::Error> {
        classifier_builder_check!(self);
        let buf = base_task_builder_check_and_get_buf!(self);

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model::parse_model(buf.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

        // check model
        model_base_check_impl!(model_resource, 1, 4);
        let _img_info = model_resource_check_and_get_impl!(model_resource, image_to_tensor_info, 0);

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_builder.execution_target,
        )
        .build_from_shared_slices([buf])?;

        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);
        let location_buf_index = model_resource_check_and_get_impl!(
            model_resource,
            output_tensor_name_to_index,
            "location"
        );
        let mut bound_box_properties = [0, 1, 2, 3];
        if model_resource
            .output_bounding_box_properties(location_buf_index, &mut bound_box_properties)
        {
            for i in 0..4 {
                if bound_box_properties[i] >= 4 {
                    return Err(crate::Error::ModelInconsistentError(format!(
                        "BoundingBoxProperties must contains `0,1,2,3`, but got `{}`",
                        bound_box_properties[i]
                    )));
                }
            }
        }

        let categories_buf_index = model_resource_check_and_get_impl!(
            model_resource,
            output_tensor_name_to_index,
            "category"
        );
        let score_buf_index = model_resource_check_and_get_impl!(
            model_resource,
            output_tensor_name_to_index,
            "score"
        );
        let num_box_buf_index = {
            let mut p = [true; 4];
            p[location_buf_index] = false;
            p[categories_buf_index] = false;
            p[score_buf_index] = false;
            let mut i = 0;
            while i < 4 {
                if p[i] {
                    break;
                }
                i += 1;
            }
            i
        };
        return Ok(ObjectDetector {
            build_info: self,
            model_resource,
            graph,
            bound_box_properties,
            location_buf_index,
            categories_buf_index,
            score_buf_index,
            num_box_buf_index,
            input_tensor_type,
        });
    }
}
