macro_rules! detector_impl {
    ( $DetectorSessionName:ident, $Result:ident ) => {
        base_task_options_get_impl!();

        /// Detect one image.
        #[inline(always)]
        pub fn detect(
            &self,
            input: &impl crate::preprocess::vision::ImageToTensor,
        ) -> Result<$Result, crate::Error> {
            self.new_session()?.detect(input)
        }

        /// Detect input video stream, and collect all results to [`Vec`]
        #[inline(always)]
        pub fn detect_for_video(
            &self,
            video_data: impl crate::preprocess::vision::VideoData,
        ) -> Result<Vec<$Result>, crate::Error> {
            self.new_session()?.detect_for_video(video_data)?.to_vec()
        }
    };
}

macro_rules! detector_session_impl {
    ( $Result:ident ) => {
        /// Detect one image
        #[inline(always)]
        pub fn detect(
            &mut self,
            input: &impl crate::preprocess::vision::ImageToTensor,
        ) -> Result<$Result, crate::Error> {
            input.to_tensor(self.image_to_tensor_info, &mut self.input_buffer)?;
            self.compute(input.time_stamp_ms())
        }

        /// Detect input video stream use this session.
        /// Return a iterator for results, process input stream when poll next result.
        #[inline(always)]
        pub fn detect_for_video<InputVideoData: crate::preprocess::vision::VideoData>(
            &mut self,
            video_data: InputVideoData,
        ) -> Result<crate::postprocess::VideoResultsIter<Self, InputVideoData>, crate::Error> {
            Ok(crate::postprocess::VideoResultsIter::new(self, video_data))
        }
    };
}

macro_rules! detection_task_session_impl {
    ( $SessionName:ident, $Result:ident ) => {
        use crate::preprocess::vision::ImageToTensor;

        impl<'model> super::TaskSession for $SessionName<'model> {
            type Result = $Result;

            #[inline]
            fn process_next(
                &mut self,
                video_data: &mut impl crate::preprocess::vision::VideoData,
            ) -> Result<Option<Self::Result>, crate::Error> {
                if let Some(frame) = video_data.next_frame()? {
                    frame.to_tensor(self.image_to_tensor_info, &mut self.input_buffer)?;
                    return Ok(Some(self.compute(frame.time_stamp_ms())?));
                }
                Ok(None)
            }
        }
    };
}

macro_rules! get_type_and_quantization {
    ( $self:ident, $index:expr ) => {{
        let t =
            model_resource_check_and_get_impl!($self.model_resource, output_tensor_type, $index);
        let q = $self
            .model_resource
            .output_tensor_quantization_parameters($index);
        check_quantization_parameters!(t, q, $index);

        (t, q)
    }};
}
