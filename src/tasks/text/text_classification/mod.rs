mod builder;
pub use builder::TextClassifierBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::sessions::{CategoriesFilter, ClassificationSession};
use crate::postprocess::ClassificationResult;
use crate::preprocess::Tensor;
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on text.
pub struct TextClassifier {
    build_info: TextClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
}

impl TextClassifier {
    base_task_build_info_get_impl!();

    classifier_build_info_get_impl!();

    #[inline(always)]
    pub fn new_session(&self) -> Result<TextClassifierSession, Error> {
        let input_count = self.model_resource.input_tensor_count();
        let mut input_tensor_shapes = Vec::with_capacity(input_count);
        let mut input_tensor_bufs = Vec::with_capacity(input_count);
        for i in 0..input_count {
            let input_tensor_shape =
                model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, i);
            let bytes = input_tensor_shape.iter().fold(4, |sum, b| sum * *b);
            input_tensor_shapes.push(input_tensor_shape);
            input_tensor_bufs.push(vec![0; bytes]);
        }

        let output_byte_size =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_byte_size, 0);
        let output_tensor_type =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_type, 0);

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
        classification_session.add_output_cfg(vec![0; output_byte_size], output_tensor_type, None);

        Ok(TextClassifierSession {
            classifier: self,
            execution_ctx,
            classification_session,
            input_tensor_shapes,
            input_tensor_bufs,
        })
    }

    #[inline(always)]
    pub fn classify(&self, input: &impl Tensor) -> Result<ClassificationResult, Error> {
        self.new_session()?.classify(input)
    }
}

/// Session to run inference. If process multiple text, use it can get better performance.
///
/// ```rust
/// use mediapipe_rs::tasks::text::TextClassifier;
///
/// let text_classifier: TextClassifier;
/// let mut session = text_classifier.new_session()?;
/// for text in texts {
///     session.classify(text)?;
/// }
/// ```
pub struct TextClassifierSession<'a> {
    classifier: &'a TextClassifier,
    execution_ctx: GraphExecutionContext<'a>,
    classification_session: ClassificationSession<'a>,

    input_tensor_shapes: Vec<&'a [usize]>,
    input_tensor_bufs: Vec<Vec<u8>>,
}

impl<'a> TextClassifierSession<'a> {
    pub fn classify(&mut self, input: &impl Tensor) -> Result<ClassificationResult, Error> {
        let to_tensor_info =
            model_resource_check_and_get_impl!(self.classifier.model_resource, to_tensor_info, 0);
        let mut input_buffers: Vec<&mut [u8]> = self
            .input_tensor_bufs
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect();
        input.to_tensors(to_tensor_info, input_buffers.as_mut_slice())?;

        for index in 0..self.input_tensor_bufs.len() {
            self.execution_ctx.set_input(
                index,
                TensorType::I32,
                self.input_tensor_shapes[index],
                self.input_tensor_bufs[index].as_slice(),
            )?;
        }
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
