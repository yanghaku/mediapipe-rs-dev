macro_rules! detector_impl {
    ( $DetectorSessionName:ident, $Result:ident ) => {
        base_task_options_get_impl!();

        /// Detect one image.
        #[inline(always)]
        pub fn detect(&self, input: &impl crate::preprocess::Tensor) -> Result<$Result, crate::Error> {
            self.new_session()?.detect(input)
        }

        /// Detect input video stream, and collect all results to [`Vec`]
        #[inline(always)]
        pub fn detect_for_video<'a>(
            &'a self,
            input_stream: impl crate::preprocess::InToTensorsIterator<'a>,
        ) -> Result<Vec<$Result>, crate::Error> {
            let iter = self.detection_results_iter(input_stream)?;
            let mut session = self.new_session()?;
            iter.to_vec(&mut session)
        }

        /// Return a iterator for results, process input stream when poll next result.
        #[inline(always)]
        pub fn detection_results_iter<'a, T>(
            &'a self,
            input_stream: T,
        ) -> Result<crate::postprocess::ResultsIter<$DetectorSessionName<'_>, T::Iter>, crate::Error>
        where
            T: crate::preprocess::InToTensorsIterator<'a>,
        {
            let to_tensor_info =
                model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0);
            let input_tensors_iter = input_stream.into_tensors_iter(to_tensor_info)?;
            Ok(crate::postprocess::ResultsIter::new(input_tensors_iter))
        }
    };
}

macro_rules! detector_session_impl {
    ( $Result:ident ) => {
        /// Detect one image
        #[inline(always)]
        pub fn detect(
            &mut self,
            input: &impl crate::preprocess::Tensor,
        ) -> Result<$Result, crate::Error> {
            let to_tensor_info =
                model_resource_check_and_get_impl!(self.detector.model_resource, to_tensor_info, 0);
            input.to_tensors(to_tensor_info, &mut [&mut self.input_buffer])?;
            self.compute(None)
        }

        /// Detect input video stream use this session, and collect all results to [`Vec`]
        #[inline(always)]
        pub fn detect_for_video(
            &'a mut self,
            input_stream: impl crate::preprocess::InToTensorsIterator<'a>,
        ) -> Result<Vec<$Result>, crate::Error> {
            self.detector
                .detection_results_iter(input_stream)?
                .to_vec(self)
        }
    };
}

macro_rules! detection_task_session_impl {
    ( $SessionName:ident, $Result:ident ) => {
        impl<'a> crate::tasks::TaskSession for $SessionName<'a> {
            type Result = $Result;

            #[inline]
            fn process_next<TensorsIter: crate::preprocess::TensorsIterator>(
                &mut self,
                input_tensors_iter: &mut TensorsIter,
            ) -> Result<Option<Self::Result>, crate::Error> {
                if let Some(timestamp_ms) =
                    input_tensors_iter.next_tensors(&mut [&mut self.input_buffer])?
                {
                    return Ok(Some(self.compute(Some(timestamp_ms))?));
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
