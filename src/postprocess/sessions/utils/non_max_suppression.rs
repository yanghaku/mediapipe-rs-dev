use crate::postprocess::{Detection, DetectionResult, Rect};

#[derive(Debug, Copy, Clone)]
pub enum NonMaxSuppressionOverlapType {
    Jaccard,
    ModifiedJaccard,
    IntersectionOverUnion,
}

#[derive(Debug, Copy, Clone)]
pub enum NonMaxSuppressionAlgorithm {
    DEFAULT,
    WEIGHTED,
}

pub struct NonMaxSuppressionBuilder {
    overlap_type: NonMaxSuppressionOverlapType,
    algorithm: NonMaxSuppressionAlgorithm,
    max_results: usize,
    min_suppression_threshold: f32,
    min_score_threshold: f32,
}

impl NonMaxSuppressionBuilder {
    #[inline(always)]
    pub fn new(max_results: i32, min_score_threshold: f32) -> Self {
        let max_results = if max_results < 0 {
            usize::MAX
        } else {
            max_results as usize
        };
        Self {
            overlap_type: NonMaxSuppressionOverlapType::Jaccard,
            algorithm: NonMaxSuppressionAlgorithm::DEFAULT,
            max_results,
            min_suppression_threshold: 1.0, // default
            min_score_threshold,
        }
    }

    #[inline(always)]
    pub fn overlap_type(mut self, overlap_type: NonMaxSuppressionOverlapType) -> Self {
        self.overlap_type = overlap_type;
        self
    }

    #[inline(always)]
    pub fn algorithm(mut self, algorithm: NonMaxSuppressionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    #[inline(always)]
    pub fn max_results(mut self, max_results: i32) -> Self {
        self.max_results = if max_results < 0 {
            usize::MAX
        } else {
            max_results as usize
        };
        self
    }

    #[inline]
    pub fn do_nms(&self, detection_result: &mut DetectionResult) {
        // remove all but the maximum scoring label from each input detection.
        detection_result
            .detections
            .retain(|d| !d.categories.is_empty());
        for d in &mut detection_result.detections {
            if d.categories.len() > 1 {
                d.categories.sort();
                d.categories.drain(1..);
            }
        }

        let mut indexed_scores = Vec::with_capacity(detection_result.detections.len());
        for i in 0..detection_result.detections.len() {
            indexed_scores.push((i, detection_result.detections[i].categories[0].score));
        }
        indexed_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        match self.algorithm {
            NonMaxSuppressionAlgorithm::DEFAULT => {
                self.non_max_suppression(&mut detection_result.detections, indexed_scores);
            }
            NonMaxSuppressionAlgorithm::WEIGHTED => {
                todo!("")
            }
        }
    }

    fn non_max_suppression(
        &self,
        detections: &mut Vec<Detection>,
        indexed_scores: Vec<(usize, f32)>,
    ) {
        let mut retains = vec![false; detections.len()];
        let mut retained_locations = Vec::new();
        for (index, score) in indexed_scores {
            if self.min_score_threshold > 0. && score < self.min_score_threshold {
                break;
            }
            let location = &detections[index].bounding_box;
            let mut suppressed = false;

            for retained_location in &retained_locations {
                let similarity = self.overlap_similarity(location, *retained_location);
                if similarity > self.min_suppression_threshold {
                    suppressed = true;
                    break;
                }
            }

            if !suppressed {
                retains[index] = true;
                retained_locations.push(location);
                if retained_locations.len() >= self.max_results {
                    break;
                }
            }
        }

        let mut iter = retains.iter();
        detections.retain(|_| *iter.next().unwrap());
    }

    #[inline]
    fn overlap_similarity(&self, rect_1: &Rect<f32>, rect_2: &Rect<f32>) -> f32 {
        if let Some(intersection) = rect_1.intersect(rect_2) {
            let intersection_area = intersection.area();
            let normalization = match self.overlap_type {
                NonMaxSuppressionOverlapType::Jaccard => rect_1.union(rect_2).area(),
                NonMaxSuppressionOverlapType::ModifiedJaccard => rect_2.area(),
                NonMaxSuppressionOverlapType::IntersectionOverUnion => {
                    rect_1.area() + rect_2.area() - intersection_area
                }
            };
            if normalization > 0. {
                intersection_area / normalization
            } else {
                0.
            }
        } else {
            0.
        }
    }
}
