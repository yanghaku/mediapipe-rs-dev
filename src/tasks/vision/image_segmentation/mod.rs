mod builder;
mod result;
pub use builder::ImageSegmenterBuilder;
pub use result::ImageSegmenterResult;

use crate::model::ModelResourceTrait;
use crate::postprocess::{Activation, TensorsToSegmentation, VideoResultsIter};
use crate::preprocess::vision::{ImageToTensor, ImageToTensorInfo, VideoData};
use crate::{Error, Graph, GraphExecutionContext, TensorType};

/// Performs segmentation on images.
pub struct ImageSegmenter {
    build_options: ImageSegmenterBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    labels: Vec<String>,
    labels_locale: Option<Vec<String>>,

    input_tensor_type: TensorType,
    output_activation: Activation,
}

impl ImageSegmenter {
    base_task_options_get_impl!();

    /// Get display names locale
    #[inline(always)]
    pub fn display_names_locale(&self) -> &String {
        &self.build_options.display_names_locale
    }

    #[inline(always)]
    pub fn output_category_mask(&self) -> bool {
        self.build_options.output_category_mask
    }

    #[inline(always)]
    pub fn output_confidence_masks(&self) -> bool {
        self.build_options.output_confidence_masks
    }

    /// get labels for the task model
    #[inline(always)]
    pub fn labels(&self) -> &Vec<String> {
        &self.labels
    }

    /// get locale labels for the task model
    #[inline(always)]
    pub fn labels_locale(&self) -> &Option<Vec<String>> {
        &self.labels_locale
    }

    #[inline(always)]
    pub fn new_session(&self) -> Result<ImageSegmenterSession, Error> {
        let input_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);
        let output_shape =
            model_resource_check_and_get_impl!(self.model_resource, output_tensor_shape, 0);

        let tensors_to_segmentation = TensorsToSegmentation::new(
            self.output_activation,
            get_type_and_quantization!(self, 0),
            output_shape,
        );
        let execution_ctx = self.graph.init_execution_context()?;
        Ok(ImageSegmenterSession {
            execution_ctx,
            tensors_to_segmentation,
            input_to_tensor_info,
            input_tensor_shape,
            input_tensor_buf: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
            input_tensor_type: self.input_tensor_type,
        })
    }

    /// Segment one image.
    #[inline(always)]
    pub fn segment(&self, input: &impl ImageToTensor) -> Result<ImageSegmenterResult, Error> {
        self.new_session()?.segment(input)
    }

    /// Segment audio stream, and collect all results to [`Vec`]
    #[inline(always)]
    pub fn segment_for_video(
        &self,
        video_data: impl VideoData,
    ) -> Result<Vec<ImageSegmenterResult>, Error> {
        self.new_session()?.segment_for_video(video_data)?.to_vec()
    }
}

pub struct ImageSegmenterSession<'model> {
    execution_ctx: GraphExecutionContext<'model>,
    tensors_to_segmentation: TensorsToSegmentation,

    // only one input and one output
    input_to_tensor_info: &'model ImageToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_tensor_buf: Vec<u8>,
    input_tensor_type: TensorType,
}

impl<'model> ImageSegmenterSession<'model> {
    #[inline(always)]
    fn compute(&mut self) -> Result<ImageSegmenterResult, Error> {
        self.execution_ctx.set_input(
            0,
            self.input_tensor_type,
            self.input_tensor_shape,
            self.input_tensor_buf.as_ref(),
        )?;

        self.execution_ctx.compute()?;

        let output_buffer = self.tensors_to_segmentation.tenor_buffer();
        self.execution_ctx.get_output(0, output_buffer)?;

        todo!()
    }

    /// Segment one image, reuse this session data to speedup.
    #[inline(always)]
    pub fn segment(&mut self, input: &impl ImageToTensor) -> Result<ImageSegmenterResult, Error> {
        input.to_tensor(
            self.input_to_tensor_info,
            &Default::default(),
            &mut self.input_tensor_buf,
        )?;
        self.compute()
    }

    /// Segment input video stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn segment_for_video<InputVideoData: VideoData>(
        &mut self,
        video_data: InputVideoData,
    ) -> Result<VideoResultsIter<Self, InputVideoData>, Error> {
        Ok(VideoResultsIter::new(self, video_data))
    }
}

impl<'model> super::TaskSession for ImageSegmenterSession<'model> {
    type Result = ImageSegmenterResult;

    #[inline]
    fn process_next(
        &mut self,
        process_options: &super::ImageProcessingOptions,
        video_data: &mut impl VideoData,
    ) -> Result<Option<Self::Result>, Error> {
        if process_options.region_of_interest.is_some() {
            return Err(Error::ArgumentError(format!(
                "{} does not support region of interest.",
                stringify!($SessionName)
            )));
        }

        // todo: support rotation
        assert_eq!(process_options.rotation, 0.);

        if let Some(frame) = video_data.next_frame()? {
            frame.to_tensor(
                self.input_to_tensor_info,
                process_options,
                &mut self.input_tensor_buf,
            )?;
            return Ok(Some(self.compute()?));
        }
        Ok(None)
    }
}
