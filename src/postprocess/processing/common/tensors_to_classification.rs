use super::*;
use crate::postprocess::{ClassificationResult, Classifications};

pub(crate) struct TensorsToClassification<'a> {
    categories_filters: Vec<CategoriesFilter<'a>>,
    outputs: Vec<OutputBuffer>,
    max_results: Vec<usize>,
}

impl<'a> TensorsToClassification<'a> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            categories_filters: Vec::new(),
            outputs: Vec::new(),
            max_results: Vec::new(),
        }
    }

    pub(crate) fn add_classification_options(
        &mut self,
        categories_filter: CategoriesFilter<'a>,
        max_results: i32,
        data_buffer: Vec<u8>,
        tensor_type: TensorType,
        quantization_parameters: Option<QuantizationParameters>,
    ) {
        let max_results = if max_results < 0 {
            usize::MAX
        } else {
            max_results as usize
        };
        self.categories_filters.push(categories_filter);
        self.max_results.push(max_results);

        self.add_output_cfg(data_buffer, tensor_type, quantization_parameters);
    }

    output_buffer_impl!();

    #[inline]
    pub(crate) fn result(&mut self, timestamp_ms: Option<u64>) -> ClassificationResult {
        let classifications_count = self.outputs.len();
        let mut res = ClassificationResult {
            classifications: Vec::with_capacity(classifications_count),
            timestamp_ms,
        };

        for id in 0..classifications_count {
            let max_results = self.max_results[id];
            let categories_filter = self.categories_filters.get(id).unwrap();

            let out = self.outputs.get_mut(id).unwrap();
            let scores = output_buffer_mut_slice!(out);
            let mut categories = Vec::new();
            for i in 0..scores.len() {
                if let Some(category) = categories_filter.create_category(i, scores[i]) {
                    categories.push(category);
                }
            }

            categories.sort();
            if max_results < categories.len() {
                categories.drain(max_results..);
            }
            res.classifications.push(Classifications {
                head_index: 0,
                head_name: None,
                categories,
            })
        }

        res
    }
}
