use super::AudioClassifier;
use crate::model::ModelResourceTrait;
use crate::tasks::common::{BaseTaskOptions, ClassificationOptions};
use crate::Error;

/// Configure the properties of a new Audio classification task.
/// Methods can be chained on it in order to configure it.
pub struct AudioClassifierBuilder {
    pub(super) base_task_options: BaseTaskOptions,
    pub(super) classification_options: ClassificationOptions,
}

impl AudioClassifierBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            base_task_options: Default::default(),
            classification_options: Default::default(),
        }
    }

    base_task_options_impl!();

    classification_options_impl!();

    #[inline]
    pub fn finalize(mut self) -> Result<AudioClassifier, Error> {
        classification_options_check!(self);
        let buf = base_task_options_check_and_get_buf!(self);

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model::parse_model(buf.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

        // check model
        model_base_check_impl!(model_resource, 1, 1);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_audio()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.execution_target,
        )
        .build_from_shared_slices([buf])?;

        return Ok(AudioClassifier {
            build_options: self,
            model_resource,
            graph,
            input_tensor_type,
        });
    }
}

#[cfg(test)]
mod test {
    use crate::tasks::audio::AudioClassifierBuilder;

    #[test]
    fn test_builder_check() {
        assert!(AudioClassifierBuilder::new().finalize().is_err());
        assert!(AudioClassifierBuilder::new()
            .model_asset_buffer("".into())
            .model_asset_path("")
            .finalize()
            .is_err());
        assert!(AudioClassifierBuilder::new()
            .model_asset_path("")
            .max_results(0)
            .finalize()
            .is_err());
    }
}
