/// Quantization parameters corresponding to the zero_point and scale value.
#[derive(Debug, Copy, Clone)]
pub struct QuantizationParameters {
    pub scale: f32,
    pub zero_point: i32,
}

pub(crate) trait Dequantize {
    fn dequantize(&self, quantization_parameters: QuantizationParameters) -> Vec<f32>;
}

impl Dequantize for &[u8] {
    #[inline(always)]
    fn dequantize(&self, quantization_parameters: QuantizationParameters) -> Vec<f32> {
        let mut res = Vec::with_capacity(self.len());
        for input in self.iter() {
            let output = quantization_parameters.scale
                * (*input as i32 - quantization_parameters.zero_point) as f32;
            res.push(output)
        }
        res
    }
}
