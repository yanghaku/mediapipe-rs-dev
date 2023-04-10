macro_rules! detector_impl {
    ( $DetectorSessionName:ident, $Result:ident ) => {
        base_task_options_get_impl!();

        /// Detect one image.
        #[inline(always)]
        pub fn detect(
            &self,
            input: &impl crate::preprocess::Tensor,
        ) -> Result<$Result, crate::Error> {
            self.new_session()?.detect(input)
        }

        /// Detect input video stream, and collect all results to [`Vec`]
        #[inline(always)]
        pub fn detect_for_video<'model: 'tensor, 'tensor>(
            &'model self,
            input_stream: impl crate::preprocess::InToTensorsIterator<'tensor>,
        ) -> Result<Vec<$Result>, crate::Error> {
            self.new_session()?.detect_for_video(input_stream)?.to_vec()
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

        /// Detect input video stream use this session.
        /// Return a iterator for results, process input stream when poll next result.
        #[inline(always)]
        pub fn detect_for_video<'session, 'tensor, T>(
            &'session mut self,
            input_stream: T,
        ) -> Result<crate::postprocess::ResultsIter<'session, 'tensor, Self, T::Iter>, crate::Error>
        where
            T: crate::preprocess::InToTensorsIterator<'tensor>,
            'model: 'tensor,
        {
            let to_tensor_info =
                model_resource_check_and_get_impl!(self.detector.model_resource, to_tensor_info, 0);
            let input_tensors_iter = input_stream.into_tensors_iter(to_tensor_info)?;
            Ok(crate::postprocess::ResultsIter::new(
                self,
                input_tensors_iter,
            ))
        }
    };
}

macro_rules! detection_task_session_impl {
    ( $SessionName:ident, $Result:ident ) => {
        impl<'model> crate::tasks::TaskSession for $SessionName<'model> {
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
