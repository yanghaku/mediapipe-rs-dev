mod generated;

use super::{
    DataLayout, Error, GraphEncoding, ImageToTensorInfo, ModelResourceTrait,
    QuantizationParameters, TensorType,
};
use crate::preprocess::vision::ImageColorSpaceType;
use generated::*;

pub(crate) struct TfLiteModelResource<'buf> {
    model: tflite::Model<'buf>,
    metadata: Option<tflite_metadata::ModelMetadata<'buf>>,
    input_shape: Vec<Vec<usize>>,
    output_shape: Vec<Vec<usize>>,
    input_types: Vec<TensorType>,
    output_types: Vec<TensorType>,
    output_bytes_size: Vec<usize>,
    output_quantization_parameters: Vec<Option<QuantizationParameters>>,
    image_to_tensor_info: Vec<Option<ImageToTensorInfo>>,
}

impl<'buf> TfLiteModelResource<'buf> {
    // 1c000000 TFL3
    pub(super) const HEAD_MAGIC: &'static [u8] = &[0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33];

    const METADATA_NAME: &'static str = "TFLITE_METADATA";

    pub(super) fn new(buf: &'buf [u8]) -> Result<Self, Error> {
        let model = tflite::root_as_model(buf)?;
        let mut _self = Self {
            model,
            metadata: None,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            input_types: Vec::new(),
            output_types: Vec::new(),
            output_bytes_size: Vec::new(),
            output_quantization_parameters: Vec::new(),
            image_to_tensor_info: Vec::new(),
        };
        _self.parse_subgraph()?;
        _self.parse_model_metadata()?;
        if _self.metadata.is_some() {
            _self.parse_model_metadata_content()?;
        }
        Ok(_self)
    }

    #[inline]
    fn parse_subgraph(&mut self) -> Result<(), Error> {
        let subgraph = match self.model.subgraphs() {
            Some(s) => {
                if s.len() < 1 {
                    return Err(Error::ModelParseError("Model subgraph is empty".into()));
                }
                s.get(0)
            }
            None => {
                return Err(Error::ModelParseError(format!("Model has no subgraph")));
            }
        };

        if let (Some(inputs), Some(outputs), Some(tensors)) =
            (subgraph.inputs(), subgraph.outputs(), subgraph.tensors())
        {
            self.input_shape = Vec::with_capacity(inputs.len());
            self.input_types = Vec::with_capacity(inputs.len());
            for i in 0..inputs.len() {
                let index = inputs.get(i) as usize;
                if index >= tensors.len() {
                    return Err(Error::ModelParseError(format!(
                        "Invalid tensor input: index `{}` larger than tensor number `{}`",
                        index,
                        tensors.len()
                    )));
                }
                let t = tensors.get(index);
                self.input_types.push(Self::tflite_type_parse(t.type_())?);
                if let Some(s) = t.shape() {
                    let len = s.len();
                    let mut shape = Vec::with_capacity(len);
                    for d in 0..len {
                        shape.push(s.get(d) as usize);
                    }
                    self.input_shape.push(shape);
                } else {
                    return Err(Error::ModelParseError(format!(
                        "Missing tensor shape for input `{}`",
                        i
                    )));
                }
            }

            self.output_shape = Vec::with_capacity(outputs.len());
            self.output_types = Vec::with_capacity(outputs.len());
            for i in 0..outputs.len() {
                let index = outputs.get(i) as usize;
                if index >= tensors.len() {
                    return Err(Error::ModelParseError(format!(
                        "Invalid tensor output: index `{}` larger than tensor number `{}`",
                        index,
                        tensors.len()
                    )));
                }
                let t = tensors.get(index);
                let tensor_type = Self::tflite_type_parse(t.type_())?;
                let mut bytes = tensor_byte_size!(tensor_type);
                self.output_types.push(tensor_type);

                if let Some(s) = t.shape() {
                    let len = s.len();
                    let mut shape = Vec::with_capacity(len);
                    for d in 0..len {
                        let val = s.get(d) as usize;
                        bytes *= val;
                        shape.push(val);
                    }
                    self.output_shape.push(shape);
                } else {
                    return Err(Error::ModelParseError(format!(
                        "Missing tensor shape for output `{}`",
                        i
                    )));
                }
                self.output_bytes_size.push(bytes);

                if let Some(q) = t.quantization() {
                    if let (Some(z), Some(s)) = (q.zero_point(), q.scale()) {
                        if z.len() > 0 && s.len() > 0 {
                            while self.output_quantization_parameters.len() < i {
                                self.output_quantization_parameters.push(None);
                            }
                            self.output_quantization_parameters.push(Some(
                                QuantizationParameters {
                                    scale: s.get(0),
                                    zero_point: z.get(0) as i32,
                                },
                            ));
                        }
                    }
                }
            }
        } else {
            return Err(Error::ModelParseError(
                "Model must has inputs, outputs and tensors information.".into(),
            ));
        }
        Ok(())
    }

    #[inline]
    fn parse_model_metadata(&mut self) -> Result<(), Error> {
        if let (Some(metadata_vec), Some(model_buffers)) =
            (self.model.metadata(), self.model.buffers())
        {
            for i in 0..metadata_vec.len() {
                let m = metadata_vec.get(i);
                if m.name() == Some(Self::METADATA_NAME) {
                    let buf_index = m.buffer() as usize;
                    if buf_index < model_buffers.len() {
                        let data_option = model_buffers.get(buf_index).data();
                        if data_option.is_some() {
                            // todo: submit an issue to flatbuffers and fix the checked error in rust
                            let metadata = unsafe {
                                tflite_metadata::root_as_model_metadata_unchecked(
                                    data_option.unwrap().bytes(),
                                )
                            };
                            self.metadata = Some(metadata);
                            return Ok(());
                        }
                    }

                    return Err(Error::ModelParseError(format!(
                        "Missing model buffer (index = `{}`)",
                        buf_index
                    )));
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn parse_model_metadata_content(&mut self) -> Result<(), Error> {
        let subgraph = match self.metadata.unwrap().subgraph_metadata() {
            Some(s) => {
                if s.len() < 1 {
                    return Ok(());
                }
                s.get(0)
            }
            None => {
                return Ok(());
            }
        };
        if let Some(input_tensors) = subgraph.input_tensor_metadata() {
            let len = input_tensors.len();
            for i in 0..len {
                let input = input_tensors.get(i);
                if input.name() == Some("image") {
                    let (height, width) = if let Some(shape) = self.input_shape.get(i) {
                        if shape.len() == 4 && shape[0] == 1 && (shape[3] == 3 || shape[3] == 1) {
                            (shape[1] as u32, shape[2] as u32)
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };
                    let (stats_min, stats_max) = if let Some(stats) = input.stats() {
                        let min = if let Some(m) = stats.min() {
                            m.iter().collect()
                        } else {
                            vec![]
                        };
                        let max = if let Some(m) = stats.max() {
                            m.iter().collect()
                        } else {
                            vec![]
                        };
                        (min, max)
                    } else {
                        (vec![], vec![])
                    };

                    let mut normalization_options = (vec![], vec![]);
                    if let Some(n) = input.process_units() {
                        for i in 0..n.len() {
                            if let Some(p) = n.get(i).options_as_normalization_options() {
                                normalization_options.0 = if let Some(m) = p.mean() {
                                    m.iter().collect()
                                } else {
                                    vec![]
                                };
                                normalization_options.1 = if let Some(m) = p.std_() {
                                    m.iter().collect()
                                } else {
                                    vec![]
                                };
                                break;
                            }
                        }
                    }

                    // todo: color_space
                    let img_info = ImageToTensorInfo {
                        color_space: ImageColorSpaceType::RGB,
                        tensor_type: self.input_types.get(i).unwrap().clone(),
                        width,
                        height,
                        stats_min,
                        stats_max,
                        normalization_options,
                    };

                    while self.image_to_tensor_info.len() < i {
                        self.image_to_tensor_info.push(None);
                    }
                    self.image_to_tensor_info.push(Some(img_info));
                } else {
                    // todo
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn tflite_type_parse(tflite_type: tflite::TensorType) -> Result<TensorType, Error> {
        match tflite_type {
            tflite::TensorType::FLOAT32 => Ok(TensorType::F32),
            tflite::TensorType::UINT8 => Ok(TensorType::U8),
            tflite::TensorType::INT32 => Ok(TensorType::I32),
            tflite::TensorType::FLOAT16 => Ok(TensorType::F16),
            _ => Err(Error::ModelParseError(format!(
                "Unsupported tensor type `{:?}`",
                tflite_type
            ))),
        }
    }
}

impl<'buf> ModelResourceTrait for TfLiteModelResource<'buf> {
    fn model_backend(&self) -> GraphEncoding {
        return GraphEncoding::TensorflowLite;
    }

    fn data_layout(&self) -> DataLayout {
        DataLayout::NHWC
    }

    fn input_tensor_count(&self) -> usize {
        self.input_shape.len()
    }

    fn output_tensor_count(&self) -> usize {
        self.output_shape.len()
    }

    fn input_tensor_type(&self, index: usize) -> Option<TensorType> {
        self.input_types.get(index).cloned()
    }

    fn output_tensor_type(&self, index: usize) -> Option<TensorType> {
        self.output_types.get(index).cloned()
    }

    fn input_tensor_shape(&self, index: usize) -> Option<&[usize]> {
        self.input_shape.get(index).map(|v| v.as_slice())
    }

    fn output_tensor_shape(&self, index: usize) -> Option<&[usize]> {
        self.output_shape.get(index).map(|v| v.as_slice())
    }

    fn output_tensor_byte_size(&self, index: usize) -> Option<usize> {
        self.output_bytes_size.get(index).cloned()
    }

    fn output_tensor_quantization_parameters(
        &self,
        index: usize,
    ) -> Option<QuantizationParameters> {
        if let Some(i) = self.output_quantization_parameters.get(index) {
            i.clone()
        } else {
            None
        }
    }

    fn image_to_tensor_info(&self, input_index: usize) -> Option<&ImageToTensorInfo> {
        if let Some(i) = self.image_to_tensor_info.get(input_index) {
            Some(i.as_ref().unwrap())
        } else {
            None
        }
    }
}

// todo: The GPU backend isn't able to process int data. If the input tensor is quantized, forces the image preprocessing graph to use CPU backend.
