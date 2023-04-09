mod builder;
pub use builder::AudioClassifierBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{
    CategoriesFilter, ClassificationResult, ResultsIter, TensorsToClassification,
};
use crate::preprocess::{InToTensorsIterator, TensorsIterator};
use crate::tasks::TaskSession;
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on audio.
pub struct AudioClassifier {
    build_options: AudioClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    input_tensor_type: TensorType,
}

impl AudioClassifier {
    base_task_options_get_impl!();

    classification_options_get_impl!();

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
            self.build_options
                .classification_options
                .display_names_locale
                .as_ref(),
        )?;

        let categories_filter = CategoriesFilter::new(
            &self.build_options.classification_options,
            labels.0,
            labels.1,
        );
        let mut tensors_to_classification = TensorsToClassification::new(
            categories_filter,
            self.build_options.classification_options.max_results,
        );
        tensors_to_classification.add_output_cfg(
            vec![0; output_byte_size],
            output_tensor_type,
            quantization_parameters,
        );
        Ok(AudioClassifierSession {
            classifier: self,
            execution_ctx,
            tensors_to_classification,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }

    /// Classify audio stream, and collect all results to [`Vec`]
    #[inline(always)]
    pub fn classify<'a>(
        &'a self,
        input_stream: impl InToTensorsIterator<'a>,
    ) -> Result<Vec<ClassificationResult>, Error> {
        let iter = self.classify_results_iter(input_stream)?;
        let mut session = self.new_session()?;
        iter.to_vec(&mut session)
    }

    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn classify_results_iter<'a, T>(
        &'a self,
        input_stream: T,
    ) -> Result<ResultsIter<AudioClassifierSession<'_>, T::Iter>, Error>
    where
        T: InToTensorsIterator<'a>,
    {
        let to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0);
        let input_tensors_iter = input_stream.into_tensors_iter(to_tensor_info)?;
        Ok(ResultsIter::new(input_tensors_iter))
    }
}

/// Session to run inference.
pub struct AudioClassifierSession<'a> {
    classifier: &'a AudioClassifier,
    execution_ctx: GraphExecutionContext<'a>,
    tensors_to_classification: TensorsToClassification<'a>,

    // only one input and one output
    input_tensor_shape: &'a [usize],
    input_buffer: Vec<u8>,
}

impl<'a> AudioClassifierSession<'a> {
    /// Classify audio stream use this session, and collect all results to [`Vec`]
    #[inline(always)]
    pub fn classify(
        &'a mut self,
        input_stream: impl InToTensorsIterator<'a>,
    ) -> Result<Vec<ClassificationResult>, Error> {
        self.classifier
            .classify_results_iter(input_stream)?
            .to_vec(self)
    }
}

impl<'a> TaskSession for AudioClassifierSession<'a> {
    type Result = ClassificationResult;

    #[inline]
    fn process_next<TensorsIter: TensorsIterator>(
        &mut self,
        input_tensors_iter: &mut TensorsIter,
    ) -> Result<Option<Self::Result>, Error> {
        if let Some(timestamp_ms) =
            input_tensors_iter.next_tensors(&mut [&mut self.input_buffer])?
        {
            self.execution_ctx.set_input(
                0,
                self.classifier.input_tensor_type,
                self.input_tensor_shape,
                self.input_buffer.as_slice(),
            )?;
            self.execution_ctx.compute()?;

            let output_buffer = self.tensors_to_classification.output_buffer(0);
            let output_size = self.execution_ctx.get_output(0, output_buffer)?;
            if output_size != output_buffer.len() {
                return Err(Error::ModelInconsistentError(format!(
                    "Model output bytes size is `{}`, but got `{}`",
                    output_buffer.len(),
                    output_size
                )));
            }

            return Ok(Some(
                self.tensors_to_classification.result(Some(timestamp_ms)),
            ));
        }
        Ok(None)
    }
}
