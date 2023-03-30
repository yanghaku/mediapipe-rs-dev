mod builder;
pub use builder::AudioClassifierBuilder;

use crate::model_resource::ModelResourceTrait;
use crate::postprocess::sessions::{CategoriesFilter, ClassificationSession};
use crate::postprocess::ClassificationResult;
use crate::preprocess::{ToTensorStream, ToTensorStreamIterator};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on audio.
pub struct AudioClassifier {
    build_info: AudioClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    input_tensor_type: TensorType,
}

impl AudioClassifier {
    base_task_build_info_get_impl!();

    classifier_build_info_get_impl!();

    #[inline(always)]
    pub fn new_session(&self) -> Result<AudioClassifierSession, Error> {
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
        Ok(AudioClassifierSession {
            classifier: self,
            execution_ctx,
            classification_session,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }

    #[inline(always)]
    pub fn classify<'a>(
        &'a self,
        input: &'a impl ToTensorStream<'a>,
    ) -> Result<Vec<ClassificationResult>, Error> {
        self.new_session()?.classify(input)
    }
}

/// Session to run inference. If process multiple audios, use it can get better performance.
///
/// ```rust
/// use mediapipe_rs::tasks::audio::AudioClassifier;
///
/// let audio_classifier: AudioClassifier;
/// let mut session = audio_classifier.new_session()?;
/// session.classify(audio)?;
/// ```
pub struct AudioClassifierSession<'a> {
    classifier: &'a AudioClassifier,
    execution_ctx: GraphExecutionContext<'a>,
    classification_session: ClassificationSession<'a>,

    // only one input and one output
    input_tensor_shape: &'a [usize],
    input_buffer: Vec<u8>,
}

impl<'a> AudioClassifierSession<'a> {
    pub fn classify(
        &mut self,
        input: &'a impl ToTensorStream<'a>,
    ) -> Result<Vec<ClassificationResult>, Error> {
        let mut results = Vec::new();
        let model_resource = &self.classifier.model_resource;
        let mut tensors = input.to_tensors_stream(0, model_resource)?;
        while let Some(timestamp_ms) = tensors.next_tensors(&mut [&mut self.input_buffer]) {
            self.execution_ctx.set_input(
                0,
                self.classifier.input_tensor_type,
                self.input_tensor_shape,
                self.input_buffer.as_slice(),
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

            results.push(self.classification_session.result(Some(timestamp_ms)))
        }

        Ok(results)
    }
}
