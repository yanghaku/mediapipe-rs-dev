use super::super::ops::Dequantize;
use crate::model_resource::ModelResourceTrait;
use crate::tasks::common::ClassifierBuilder;
use crate::TensorType;
use std::fmt::{Display, Formatter};

/// Defines a single classification result.
///
/// The label maps packed into the TFLite Model Metadata [1] are used to populate
/// the 'category_name' and 'display_name' fields.
///
/// [1]: https://www.tensorflow.org/lite/convert/metadata
#[derive(Debug)]
pub struct Category {
    /// The index of the category in the classification model output.
    pub index: i32,

    /// The score for this category, e.g. (but not necessarily) a probability in \[0,1\].
    pub score: f32,

    /// The optional ID for the category, read from the label map packed in the
    /// TFLite Model Metadata if present. Not necessarily human-readable.
    pub category_name: Option<String>,

    /// The optional human-readable name for the category, read from the label map
    /// packed in the TFLite Model Metadata if present.
    pub display_name: Option<String>,
}

/// Defines classification results for a given classifier head.
#[derive(Debug)]
pub struct Classifications {
    /// The index of the classifier head (i.e. output tensor) these categories
    /// refer to. This is useful for multi-head models.
    pub head_index: usize,
    /// The optional name of the classifier head, as provided in the TFLite Model
    /// Metadata [1] if present. This is useful for multi-head models.
    ///
    /// [1]: https://www.tensorflow.org/lite/convert/metadata
    pub head_name: Option<String>,
    /// The array of predicted categories, usually sorted by descending scores,
    /// e.g. from high to low probability.
    pub categories: Vec<Category>,
}

/// Defines classification results of a model.
#[derive(Debug)]
pub struct ClassificationResult {
    /// The classification results for each head of the model.
    pub classifications: Vec<Classifications>,
    /// The optional timestamp (in milliseconds) of the start of the chunk of data
    /// corresponding to these results.
    ///
    /// This is only used for classification on time series (e.g. audio
    /// classification). In these use cases, the amount of data to process might
    /// exceed the maximum size that the model can process: to solve this, the
    /// input data is split into multiple chunks starting at different timestamps.
    pub timestamp_ms: Option<u64>,
}

impl Display for ClassificationResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ClassificationResult:")?;
        if let Some(t) = self.timestamp_ms {
            writeln!(f, "  Timestamp: {} ms", t)?;
        }

        if self.classifications.is_empty() {
            return writeln!(f, "  No Classification");
        }
        for i in 0..self.classifications.len() {
            writeln!(f, "  Classifications #{}:", i)?;
            let c = self.classifications.get(i).unwrap();
            if let Some(ref name) = c.head_name {
                writeln!(f, "    head name: {}", name)?;
                writeln!(f, "    head index: {}", c.head_index)?;
            }
            for j in 0..c.categories.len() {
                writeln!(f, "    category #{}:", j)?;
                let category = c.categories.get(j).unwrap();
                if let Some(ref name) = category.category_name {
                    writeln!(f, "      category name: \"{}\"", name)?;
                } else {
                    writeln!(f, "      category name: None")?;
                }
                if let Some(ref name) = category.display_name {
                    writeln!(f, "      category name: \"{}\"", name)?;
                } else {
                    writeln!(f, "      category name: None")?;
                }
                writeln!(f, "      score: {}", category.score)?;
                writeln!(f, "      index: {}", category.index)?;
            }
        }
        writeln!(f, "")
    }
}

impl ClassificationResult {
    pub(crate) fn new(
        output_index: usize,
        output_buffer: &[u8],
        model_resource: &Box<dyn ModelResourceTrait>,
        options: &ClassifierBuilder,
    ) -> Self {
        let mut categories = match model_resource.output_tensor_type(output_index).unwrap() {
            TensorType::U8 => {
                if let Some(q) = model_resource.output_tensor_quantization_parameters(output_index)
                {
                    let f = output_buffer.dequantize(q);
                    let mut c = Vec::with_capacity(f.len());
                    for i in 0..f.len() {
                        c.push(Category {
                            score: f[i],
                            index: i as i32,
                            display_name: None,
                            category_name: None,
                        })
                    }
                    c
                } else {
                    let mut c = Vec::with_capacity(output_buffer.len());
                    for i in 0..output_buffer.len() {
                        c.push(Category {
                            score: output_buffer[i] as f32,
                            index: i as i32,
                            display_name: None,
                            category_name: None,
                        })
                    }
                    c
                }
            }
            TensorType::F32 => {
                let buf = unsafe {
                    core::slice::from_raw_parts(
                        output_buffer.as_ptr() as *const f32,
                        output_buffer.len() >> 2,
                    )
                };
                let mut c = Vec::with_capacity(buf.len());
                for i in 0..buf.len() {
                    c.push(Category {
                        score: buf[i],
                        index: i as i32,
                        display_name: None,
                        category_name: None,
                    })
                }
                c
            }
            _ => unimplemented!(),
        };

        categories.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        if options.max_results > 0 {
            let max_results = options.max_results as usize;
            if max_results < categories.len() {
                categories.drain(max_results as usize..);
            }
        }

        Self {
            classifications: vec![Classifications {
                head_index: 0,
                head_name: None,
                categories,
            }],
            timestamp_ms: None,
        }
    }
}
