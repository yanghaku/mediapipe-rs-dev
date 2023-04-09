#![allow(unused)]

use crate::postprocess::ops::*;
use crate::TensorType;

struct OutputBuffer {
    data_buffer: Vec<u8>,
    tensor_type: TensorType,
    quantization_parameters: Option<(QuantizationParameters, Vec<f32>)>,
}

macro_rules! output_buffer_mut_slice {
    ( $out:expr ) => {
        match $out.tensor_type {
            TensorType::U8 => {
                let (q, f) = $out.quantization_parameters.as_mut().unwrap();
                $out.data_buffer.as_slice().dequantize_to_buf(*q, f);
                f.as_mut_slice()
            }
            TensorType::F32 => unsafe {
                core::slice::from_raw_parts_mut(
                    $out.data_buffer.as_mut_slice().as_ptr() as *mut f32,
                    $out.data_buffer.len() >> 2,
                )
            },
            _ => {
                todo!("FP16, I32")
            }
        }
    };
}

macro_rules! empty_output_buffer {
    ( $x:ident ) => {
        match $x.1 {
            Some(q) => OutputBuffer {
                data_buffer: vec![],
                tensor_type: $x.0,
                quantization_parameters: Some((q, vec![])),
            },
            None => OutputBuffer {
                data_buffer: vec![],
                tensor_type: $x.0,
                quantization_parameters: None,
            },
        }
    };
}

macro_rules! realloc_output_buffer {
    ( $self:expr, $new_size:expr ) => {
        let new_size = $new_size;
        if let Some(ref mut t) = $self.quantization_parameters {
            if t.1.len() < new_size {
                t.1.resize(new_size, 0f32);
            }
        }
        let s = tensor_byte_size!($self.tensor_type) * new_size;
        if $self.data_buffer.len() < s {
            $self.data_buffer.resize(s, 0);
        }
    };
}

macro_rules! output_buffer_impl {
    () => {
        /// index must be valid. or panic!
        #[inline(always)]
        pub(crate) fn output_buffer(&mut self, index: usize) -> &mut [u8] {
            self.outputs
                .get_mut(index)
                .unwrap()
                .data_buffer
                .as_mut_slice()
        }

        #[inline(always)]
        pub(crate) fn add_output_cfg(
            &mut self,
            data_buffer: Vec<u8>,
            tensor_type: TensorType,
            quantization_parameters: Option<QuantizationParameters>,
        ) {
            let q = if let Some(p) = quantization_parameters {
                Some((p, vec![0f32; data_buffer.len()]))
            } else {
                None
            };
            self.outputs.push(OutputBuffer {
                data_buffer,
                tensor_type,
                quantization_parameters: q,
            });
        }
    };
}

mod common;
pub(crate) use common::*;

#[cfg(feature = "vision")]
mod vision;
#[cfg(feature = "vision")]
pub(crate) use vision::*;
