use super::*;
use crate::tasks::common::ClassifierBuilder;

pub(crate) struct ClassificationSession<'a> {
    categories_filter: CategoriesFilter<'a>,
    outputs: Vec<OutputBuffer>,
    max_results: usize,
}

impl<'a> ClassificationSession<'a> {
    #[inline(always)]
    pub(crate) fn new(categories_filter: CategoriesFilter<'a>, max_results: i32) -> Self {
        let max_results = if max_results < 0 {
            usize::MAX
        } else {
            max_results as usize
        };
        Self {
            categories_filter,
            outputs: vec![],
            max_results,
        }
    }

    output_buffer_impl!();

    #[inline]
    pub(crate) fn result(&mut self, timestamp_ms: Option<u64>) -> ClassificationResult {
        // todo: multi head classify
        let classifications_count = self.outputs.len();
        let mut res = ClassificationResult {
            classifications: Vec::with_capacity(classifications_count),
            timestamp_ms,
        };

        for id in 0..classifications_count {
            let out = self.outputs.get_mut(id).unwrap();
            let scores = output_buffer_mut_slice!(out);
            let mut categories = Vec::new();
            for i in 0..scores.len() {
                if let Some(category) = self.categories_filter.new_category(i, scores[i]) {
                    categories.push(category);
                }
            }

            categories.sort();
            if self.max_results < categories.len() {
                categories.drain(self.max_results as usize..);
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
