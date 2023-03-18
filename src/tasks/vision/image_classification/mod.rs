mod builder;

use crate::model_resource::ModelResourceTrait;
use crate::postprocess::ClassificationResult;
use crate::preprocess::ToTensor;
use crate::{Error, Graph, GraphExecutionContext};

pub use builder::ImageClassifierBuilder;

/// Performs classification on images.
pub struct ImageClassifier {
    build_info: ImageClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
}

impl ImageClassifier {
    base_task_build_info_get_impl!();

    classifier_build_info_get_impl!();

    #[inline(always)]
    pub fn new_session(&self) -> Result<ImageClassifierSession, Error> {
        let output_byte_size =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_byte_size, 0);

        let execution_ctx = self.graph.init_execution_context()?;
        Ok(ImageClassifierSession {
            classifier: self,
            execution_ctx,
            output_buffer: vec![0; output_byte_size],
        })
    }

    #[inline(always)]
    pub fn classify<'t>(self, input: &impl ToTensor<'t>) -> Result<ClassificationResult, Error> {
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
    output_buffer: Vec<u8>,
}

impl<'a> ImageClassifierSession<'a> {
    pub fn classify<'t>(
        &mut self,
        input: &impl ToTensor<'t>,
    ) -> Result<ClassificationResult, Error> {
        let tensor = input.to_tensor(0, &self.classifier.model_resource)?;
        let tensor_type = model_resource_check_and_get_impl!(
            self.classifier.model_resource,
            input_tensor_type,
            0
        );
        let tensor_shape = model_resource_check_and_get_impl!(
            self.classifier.model_resource,
            input_tensor_shape,
            0
        );

        self.execution_ctx
            .set_input(0, tensor_type, tensor_shape, tensor.as_ref())?;
        self.execution_ctx.compute()?;

        let output_size = self.execution_ctx.get_output(0, &mut self.output_buffer)?;
        if output_size != self.output_buffer.len() {
            return Err(Error::ModelInconsistentError(format!(
                "Model output bytes size is `{}`, but got `{}`",
                self.output_buffer.len(),
                output_size
            )));
        }

        Ok(ClassificationResult::new(
            0,
            self.output_buffer.as_slice(),
            &self.classifier.model_resource,
            &self.classifier.build_info.classifier_builder,
        ))
    }
}
