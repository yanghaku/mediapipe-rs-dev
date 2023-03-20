use super::*;
use crate::tasks::common::ClassifierBuilder;

pub(crate) struct ClassificationSession<'a> {
    options: &'a ClassifierBuilder,
    outputs: Vec<OutputBuffer>,
}

impl<'a> ClassificationSession<'a> {
    #[inline(always)]
    pub(crate) fn new(options: &'a ClassifierBuilder) -> Self {
        Self {
            options,
            outputs: vec![],
        }
    }

    output_buffer_impl!();

    #[inline]
    pub(crate) fn result(&mut self) -> ClassificationResult {
        // todo: allow_list, timestamp, label file, multi head classify
        let classifications_count = self.outputs.len();
        let mut res = ClassificationResult {
            classifications: Vec::with_capacity(classifications_count),
            timestamp_ms: None,
        };
        let score_threshold = self.options.score_threshold;
        let max_results = if self.options.max_results < 0 {
            usize::MAX
        } else {
            self.options.max_results as usize
        };

        for id in 0..classifications_count {
            let scores = self.get_mut_slice(id);
            let mut categories = Vec::new();
            for i in 0..scores.len() {
                let score = scores[i];
                if score >= score_threshold {
                    categories.push(Category {
                        score,
                        index: i as i32,
                        display_name: None,
                        category_name: None,
                    });
                }
            }

            categories.sort();
            if max_results < categories.len() {
                categories.drain(max_results as usize..);
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
