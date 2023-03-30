use crate::postprocess::QuantizationParameters;
use crate::preprocess::vision::DataLayout;
use crate::preprocess::{AudioToTensorInfo, ImageToTensorInfo};
use crate::{Error, GraphEncoding, TensorType};

/// Abstraction for model resources.
/// Users can use this trait to get information for models, such as data layout, model backend, etc.
/// Now it supports ```TensorFlowLite``` backend.
pub trait ModelResourceTrait {
    fn model_backend(&self) -> GraphEncoding;

    fn data_layout(&self) -> DataLayout;

    fn input_tensor_count(&self) -> usize;

    fn output_tensor_count(&self) -> usize;

    fn input_tensor_type(&self, index: usize) -> Option<TensorType>;

    fn output_tensor_type(&self, index: usize) -> Option<TensorType>;

    fn input_tensor_shape(&self, index: usize) -> Option<&[usize]>;

    fn output_tensor_shape(&self, index: usize) -> Option<&[usize]>;

    fn output_tensor_byte_size(&self, index: usize) -> Option<usize>;

    fn output_tensor_name_to_index(&self, name: &'static str) -> Option<usize>;

    fn output_tensor_quantization_parameters(&self, index: usize)
        -> Option<QuantizationParameters>;

    fn output_tensor_labels_locale(
        &self,
        index: usize,
        locale: &str,
    ) -> Result<(&[u8], Option<&[u8]>), Error>;

    fn output_bounding_box_properties(&self, index: usize, slice: &mut [usize]) -> bool;

    fn image_to_tensor_info(&self, input_index: usize) -> Option<&ImageToTensorInfo>;

    fn audio_to_tensor_info(&self, input_index: usize) -> Option<&AudioToTensorInfo>;
}

#[inline]
pub(crate) fn parse_model<'buf>(
    buf: &'buf [u8],
) -> Result<Box<dyn ModelResourceTrait + 'buf>, Error> {
    if buf.len() < 8 {
        return Err(Error::ModelParseError(format!(
            "Model buffer is tool short!"
        )));
    }

    match &buf[4..8] {
        tflite::TfLiteModelResource::HEAD_MAGIC => {
            let tf_model_resource = tflite::TfLiteModelResource::new(buf)?;
            Ok(Box::new(tf_model_resource))
        }
        _ => Err(Error::ModelParseError(format!(
            "Cannot parse this head magic `{:?}`",
            &buf[..8]
        ))),
    }
}

macro_rules! model_resource_check_and_get_impl {
    ( $model_resource:expr, $func_name:ident, $index:expr ) => {
        $model_resource
            .$func_name($index)
            .ok_or(crate::Error::ModelInconsistentError(format!(
                "Model resource has no information for `{}` at index `{}`.",
                stringify!($func_name),
                $index
            )))?
    };
}

macro_rules! tensor_byte_size {
    ($tensor_type:expr) => {
        match $tensor_type {
            crate::TensorType::F32 => 4,
            crate::TensorType::U8 => 1,
            crate::TensorType::I32 => 4,
            crate::TensorType::F16 => 2,
        }
    };
}

macro_rules! tensor_bytes {
    ( $tensor_type:expr, $tensor_shape:ident ) => {{
        let mut b = tensor_byte_size!($tensor_type);
        for s in $tensor_shape {
            b *= s;
        }
        b
    }};
}

macro_rules! check_quantization_parameters {
    ( $tensor_type:ident, $q:ident, $i:expr ) => {
        if $tensor_type == crate::TensorType::U8 && $q.is_none() {
            return Err(crate::Error::ModelInconsistentError(format!(
                "Missing tensor quantization parameters for output `{}`",
                $i
            )));
        }
    };
}

mod tflite;
mod zip;
