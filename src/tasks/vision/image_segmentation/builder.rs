use super::ImageSegmenter;
use crate::model::{MemoryTextFile, ModelResourceTrait};
use crate::tasks::common::BaseTaskOptions;

/// Configure the properties of a new image segmentation task.
/// Methods can be chained on it in order to configure it.
///
/// default options:
/// * display_names_locale: "en"
/// * output_category_mask: true
/// * output_confidence_masks: false
pub struct ImageSegmenterBuilder {
    pub(super) base_task_options: BaseTaskOptions,

    /// The locale to use for display names specified through the TFLite Model
    /// Metadata, if any. Defaults to English.
    pub(super) display_names_locale: String,

    /// If set category_mask, segmentation mask will contain a uint8 image,
    /// where each pixel value indicates the winning category index.
    /// Default is true
    pub(super) output_category_mask: bool,

    /// If set confidence_masks, the segmentation masks are float images,
    /// where each float image represents the confidence score map of the category.
    /// Default is false
    pub(super) output_confidence_masks: bool,
}

impl Default for ImageSegmenterBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            display_names_locale: "en".into(),
            output_category_mask: true,
            output_confidence_masks: false,
        }
    }
}

impl ImageSegmenterBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Default::default()
    }

    base_task_options_impl!();

    /// The locale to use for display names specified through the TFLite Model
    /// Metadata, if any. Defaults to English.
    #[inline(always)]
    pub fn display_names_locale(mut self, locale: String) -> Self {
        self.display_names_locale = locale;
        self
    }

    /// Set output to category_mask.
    /// Segmentation mask will contain a uint8 image, where each pixel value indicates the winning category index.
    #[inline(always)]
    pub fn output_category_mask(mut self, output_category_mask: bool) -> Self {
        self.output_category_mask = output_category_mask;
        self
    }

    /// Set output to confidence_masks. The segmentation masks are float images,
    /// where each float image represents the confidence score map of the category.
    #[inline(always)]
    pub fn output_confidence_masks(mut self, output_confidence_masks: bool) -> Self {
        self.output_confidence_masks = output_confidence_masks;
        self
    }

    #[inline]
    pub fn finalize(mut self) -> Result<ImageSegmenter, crate::Error> {
        if !self.output_category_mask && !self.output_confidence_masks {
            return Err(crate::Error::ArgumentError(
                "At least one of the `output_category_mask` and `output_confidence_masks` be set."
                    .into(),
            ));
        }

        let buf = base_task_options_check_and_get_buf!(self);

        // change the lifetime to 'static, because the buf will move to graph and will not be released.
        let model_resource_ref = crate::model::parse_model(buf.as_ref())?;
        let model_resource = unsafe {
            std::mem::transmute::<_, Box<dyn ModelResourceTrait + 'static>>(model_resource_ref)
        };

        // check model
        model_base_check_impl!(model_resource, 1, 1);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.execution_target,
        )
        .build_from_shared_slices([buf])?;

        let (label, label_locale) =
            model_resource.output_tensor_labels_locale(0, self.display_names_locale.as_str())?;
        let mut text = MemoryTextFile::new(label);
        let mut labels = Vec::new();
        while let Some(l) = text.next_line() {
            labels.push(l.into())
        }

        let labels_locale = if let Some(f) = label_locale {
            let mut text = MemoryTextFile::new(f);
            let mut labels = Vec::new();
            while let Some(l) = text.next_line() {
                labels.push(l.into());
            }
            Some(labels)
        } else {
            None
        };

        let output_activation = model_resource.output_activation();
        return Ok(ImageSegmenter {
            build_options: self,
            model_resource,
            graph,
            labels,
            labels_locale,
            input_tensor_type,
            output_activation,
        });
    }
}
