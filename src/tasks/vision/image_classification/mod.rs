mod builder;
pub use builder::ImageClassifierBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::sessions::{CategoriesFilter, ClassificationSession};
use crate::postprocess::ClassificationResult;
use crate::preprocess::ToTensor;
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on images.
pub struct ImageClassifier {
    build_info: ImageClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    input_tensor_type: TensorType,
}

impl ImageClassifier {
    base_task_build_info_get_impl!();

    classifier_build_info_get_impl!();

    #[inline(always)]
    pub fn new_session(&self) -> Result<ImageClassifierSession, Error> {
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);
        let output_byte_size =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_byte_size, 0);
        let output_tensor_type =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_type, 0);
        let quantization_parameters = self.model_resource.output_tensor_quantization_parameters(0);
        check_quantization_parameters!(output_tensor_type, quantization_parameters, 0);

        let execution_ctx = self.graph.init_execution_context()?;
        let labels = self.model_resource.output_tensor_labels_locale(
            0,
            self.build_info
                .classifier_builder
                .display_names_locale
                .as_ref(),
        )?;

        let categories_filter =
            CategoriesFilter::new(&self.build_info.classifier_builder, labels.0, labels.1);
        let mut classification_session = ClassificationSession::new(
            categories_filter,
            self.build_info.classifier_builder.max_results,
        );
        classification_session.add_output_cfg(
            vec![0; output_byte_size],
            output_tensor_type,
            quantization_parameters,
        );
        Ok(ImageClassifierSession {
            classifier: self,
            execution_ctx,
            classification_session,
            input_tensor_shape,
            input_tensor_buf: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }

    #[inline(always)]
    pub fn classify(&self, input: &impl ToTensor) -> Result<ClassificationResult, Error> {
        self.new_session()?.classify(input)
    }
}

/// Session to run inference. If process multiple images, use it can get better performance.
///
/// ```rust
/// use mediapipe_rs::tasks::vision::ImageClassifier;
///
/// let image_classifier: ImageClassifier;
/// let mut session = image_classifier.new_session()?;
/// for image in images {
///     session.classify(image)?;
/// }
/// ```
pub struct ImageClassifierSession<'a> {
    classifier: &'a ImageClassifier,
    execution_ctx: GraphExecutionContext<'a>,
    classification_session: ClassificationSession<'a>,

    // only one input and one output
    input_tensor_shape: &'a [usize],
    input_tensor_buf: Vec<u8>,
}

impl<'a> ImageClassifierSession<'a> {
    pub fn classify(&mut self, input: &impl ToTensor) -> Result<ClassificationResult, Error> {
        input.to_tensors(
            0,
            &self.classifier.model_resource,
            &mut [&mut self.input_tensor_buf],
        )?;

        self.execution_ctx.set_input(
            0,
            self.classifier.input_tensor_type,
            self.input_tensor_shape,
            self.input_tensor_buf.as_ref(),
        )?;
        self.execution_ctx.compute()?;

        let output_buffer = self.classification_session.output_buffer(0);
        let output_size = self.execution_ctx.get_output(0, output_buffer)?;
        if output_size != output_buffer.len() {
            return Err(Error::ModelInconsistentError(format!(
                "Model output bytes size is `{}`, but got `{}`",
                output_buffer.len(),
                output_size
            )));
        }

        Ok(self.classification_session.result(None))
    }
}
