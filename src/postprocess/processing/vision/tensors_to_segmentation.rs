use super::*;
use crate::postprocess::Activation;

pub(crate) struct TensorsToSegmentation {
    activation: Activation,
    tensor_buffer: OutputBuffer,
}

impl TensorsToSegmentation {
    #[inline(always)]
    pub fn new(
        activation: Activation,
        tensor_buf_info: (TensorType, Option<QuantizationParameters>),
        tensor_shape: &[usize],
    ) -> Self {
        let elem_size = tensor_shape.iter().fold(1, |a, b| a * b);

        Self {
            activation,
            tensor_buffer: empty_output_buffer!(tensor_buf_info, elem_size),
        }
    }

    pub(crate) fn tenor_buffer(&mut self) -> &mut [u8] {
        self.tensor_buffer.data_buffer.as_mut_slice()
    }
}
