use super::ImageClassifier;
use crate::model_resource::ModelResourceTrait;
use crate::tasks::common::{BaseTaskBuilder, ClassifierBuilder};
use crate::Error;

/// Configure the properties of a new image classification task.
/// Methods can be chained on it in order to configure it.
pub struct ImageClassifierBuilder {
    pub(super) base_task_builder: BaseTaskBuilder,
    pub(super) classifier_builder: ClassifierBuilder,
}

impl ImageClassifierBuilder {
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
    pub fn finalize(mut self) -> Result<ImageClassifier, Error> {
        classifier_builder_check!(self);
        let buf = base_task_builder_check_and_get_buf!(self);

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model_resource::parse_model(buf.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

        // check model
        model_base_check_impl!(model_resource, 1, 1);
        let _ = model_resource_check_and_get_impl!(model_resource, image_to_tensor_info, 0);

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_builder.execution_target,
        )
        .build_from_shared_slices([buf])?;

        return Ok(ImageClassifier {
            build_info: self,
            model_resource,
            graph,
        });
    }
}

#[cfg(test)]
mod test {
    use crate::tasks::vision::ImageClassifierBuilder;

    #[test]
    fn test_builder_check() {
        assert!(ImageClassifierBuilder::new().finalize().is_err());
        assert!(ImageClassifierBuilder::new()
            .model_asset_buffer("".into())
            .model_asset_path("".into())
            .finalize()
            .is_err());
        assert!(ImageClassifierBuilder::new()
            .model_asset_path("".into())
            .max_results(0)
            .finalize()
            .is_err());
    }
}