mod builder;
pub use builder::ImageClassifierBuilder;

use crate::model::ModelResourceTrait;
use crate::postprocess::{
    CategoriesFilter, ClassificationResult, ResultsIter, TensorsToClassification,
};
use crate::preprocess::{InToTensorsIterator, Tensor, TensorsIterator};
use crate::tasks::TaskSession;
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs classification on images.
pub struct ImageClassifier {
    build_options: ImageClassifierBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    input_tensor_type: TensorType,
}

impl ImageClassifier {
    base_task_options_get_impl!();

    classification_options_get_impl!();

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
        Ok(ImageClassifierSession {
            classifier: self,
            execution_ctx,
            tensors_to_classification,
            input_tensor_shape,
            input_tensor_buf: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }

    /// Classify one image.
    #[inline(always)]
    pub fn classify(&self, input: &impl Tensor) -> Result<ClassificationResult, Error> {
        self.new_session()?.classify(input)
    }

    /// Classify audio stream, and collect all results to [`Vec`]
    #[inline(always)]
    pub fn classify_for_video<'model: 'tensor, 'tensor>(
        &'model self,
        input_stream: impl InToTensorsIterator<'tensor>,
    ) -> Result<Vec<ClassificationResult>, Error> {
        self.new_session()?
            .classify_for_video(input_stream)?
            .to_vec()
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
pub struct ImageClassifierSession<'model> {
    classifier: &'model ImageClassifier,
    execution_ctx: GraphExecutionContext<'model>,
    tensors_to_classification: TensorsToClassification<'model>,

    // only one input and one output
    input_tensor_shape: &'model [usize],
    input_tensor_buf: Vec<u8>,
}

impl<'model> ImageClassifierSession<'model> {
    #[inline(always)]
    fn compute(&mut self, timestamp_ms: Option<u64>) -> Result<ClassificationResult, Error> {
        self.execution_ctx.set_input(
            0,
            self.classifier.input_tensor_type,
            self.input_tensor_shape,
            self.input_tensor_buf.as_ref(),
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

        Ok(self.tensors_to_classification.result(timestamp_ms))
    }

    /// Classify one image, reuse this session data to speedup.
    #[inline(always)]
    pub fn classify(&mut self, input: &impl Tensor) -> Result<ClassificationResult, Error> {
        let to_tensor_info =
            model_resource_check_and_get_impl!(self.classifier.model_resource, to_tensor_info, 0);
        input.to_tensors(to_tensor_info, &mut [&mut self.input_tensor_buf])?;
        self.compute(None)
    }

    /// Classify input video stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn classify_for_video<'session, 'tensor, T>(
        &'session mut self,
        input_stream: T,
    ) -> Result<ResultsIter<'session, 'tensor, Self, T::Iter>, Error>
    where
        T: InToTensorsIterator<'tensor>,
        'model: 'tensor,
    {
        let to_tensor_info =
            model_resource_check_and_get_impl!(self.classifier.model_resource, to_tensor_info, 0);
        let input_tensors_iter = input_stream.into_tensors_iter(to_tensor_info)?;
        Ok(ResultsIter::new(self, input_tensors_iter))
    }
}

impl<'model> TaskSession for ImageClassifierSession<'model> {
    type Result = ClassificationResult;

    #[inline]
    fn process_next<TensorsIter: TensorsIterator>(
        &mut self,
        input_tensors_iter: &mut TensorsIter,
    ) -> Result<Option<Self::Result>, Error> {
        if let Some(timestamp_ms) =
            input_tensors_iter.next_tensors(&mut [&mut self.input_tensor_buf])?
        {
            return Ok(Some(self.compute(Some(timestamp_ms))?));
        }
        Ok(None)
    }
}
