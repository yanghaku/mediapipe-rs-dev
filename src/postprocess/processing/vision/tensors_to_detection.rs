use super::*;
use crate::postprocess::{CategoriesFilter, Category, Detection, DetectionResult, Rect};

/// Tells the calculator how to convert the detector output to bounding boxes.
#[derive(Debug, Clone, Copy)]
pub enum DetectionBoxFormat {
    /// bbox [y_center, x_center, height, width], keypoint [y, x]
    YXHW,
    /// bbox [x_center, y_center, width, height], keypoint [x, y]
    XYWH,
    /// bbox [xmin, ymin, xmax, ymax], keypoint [x, y]
    XYXY,
}

impl Default for DetectionBoxFormat {
    // if UNSPECIFIED, the calculator assumes YXHW
    fn default() -> Self {
        Self::YXHW
    }
}

macro_rules! box_y_min {
    ($box_indices:ident, $v:ident, $offset:ident) => {
        $v[$offset + $box_indices[0]]
    };
}

macro_rules! box_x_min {
    ($box_indices:ident, $v:ident, $offset:ident) => {
        $v[$offset + $box_indices[1]]
    };
}

macro_rules! box_y_max {
    ($box_indices:ident, $v:ident, $offset:ident) => {
        $v[$offset + $box_indices[2]]
    };
}

macro_rules! box_x_max {
    ($box_indices:ident, $v:ident, $offset:ident) => {
        $v[$offset + $box_indices[3]]
    };
}

pub(crate) struct TensorsToDetection<'a> {
    anchors: Option<&'a Vec<Anchor>>,
    nms: NonMaxSuppression,

    location_buf: OutputBuffer,
    score_buf: OutputBuffer,
    category_option: Option<(OutputBuffer, CategoriesFilter<'a>)>,

    /// The number of output classes predicted by the detection model.
    /// if categories buffer is not None, num_classes must be 1
    num_classes: usize,
    /// The number of output values per boxes predicted by the detection model.
    /// The values contain bounding boxes, key points, etc.
    num_coords: usize,
    /// The offset of keypoint coordinates in the location tensor.
    keypoint_coord_offset: usize,
    /// The number of predicted key points.
    num_key_points: usize, // [default = 0]
    /// The dimension of each keypoint, e.g. number of values predicted for each keypoint.
    num_values_per_key_point: usize, // [default = 2]
    /// The offset of box coordinates in the location tensor.
    box_coord_offset: usize, // [default = 0]
    /// Parameters for decoding SSD detection model.
    x_scale: f32, // [default = 0.0];
    y_scale: f32, // [default = 0.0];
    w_scale: f32, // [default = 0.0];
    h_scale: f32, // [default = 0.0];
    /// Represents the bounding box by using the combination of boundaries, {ymin, xmin, ymax, xmax}.
    /// The default order is {ymin, xmin, ymax, xmax}.
    box_indices: [usize; 4],
    box_format: DetectionBoxFormat,
    min_score_threshold: f32, // used if categories filter is none
    score_clipping_thresh: Option<f32>,
    apply_exponential_on_box_size: bool, // [default = false]
    sigmoid_score: bool,                 // [default = false];
    /// Whether the detection coordinates from the input tensors should be flipped vertically (along the y-direction).
    flip_vertically: bool, //[default = false];
}

macro_rules! check_valid {
    ( $self:ident ) => {
        assert!($self.num_coords >= $self.box_coord_offset);
        assert!(
            $self.num_coords
                >= $self.keypoint_coord_offset
                    + $self.num_key_points * $self.num_values_per_key_point
        );
    };
}

impl<'a> TensorsToDetection<'a> {
    pub(crate) fn new_with_anchors(
        anchors: &'a Vec<Anchor>,
        min_score_threshold: f32,
        max_results: i32,
        bound_box_properties: &[usize; 4],
        location_buf: (TensorType, Option<QuantizationParameters>),
        score_buf: (TensorType, Option<QuantizationParameters>),
    ) -> Self {
        Self {
            box_format: Default::default(),
            box_indices: [
                bound_box_properties[1], // y_min
                bound_box_properties[0], // x_min
                bound_box_properties[3], // y_max
                bound_box_properties[2], // x_max
            ],
            nms: NonMaxSuppression::new(max_results),
            location_buf: empty_output_buffer!(location_buf),
            score_buf: empty_output_buffer!(score_buf),
            anchors: Some(anchors),
            category_option: None,
            num_classes: 1,
            num_coords: 4,
            num_key_points: 0,
            num_values_per_key_point: 2,
            box_coord_offset: 0,
            keypoint_coord_offset: 4,
            x_scale: 0.0,
            y_scale: 0.0,
            w_scale: 0.0,
            h_scale: 0.0,
            min_score_threshold,
            score_clipping_thresh: None,
            apply_exponential_on_box_size: false,
            sigmoid_score: false,
            flip_vertically: false,
        }
    }

    #[inline]
    pub(crate) fn new(
        categories_filter: CategoriesFilter<'a>,
        max_results: i32,
        bound_box_properties: &[usize; 4],
        location_buf: (TensorType, Option<QuantizationParameters>),
        categories_buf: (TensorType, Option<QuantizationParameters>),
        score_buf: (TensorType, Option<QuantizationParameters>),
    ) -> Self {
        Self {
            box_format: Default::default(),
            box_indices: [
                bound_box_properties[1], // y_min
                bound_box_properties[0], // x_min
                bound_box_properties[3], // y_max
                bound_box_properties[2], // x_max
            ],
            nms: NonMaxSuppression::new(max_results),
            location_buf: empty_output_buffer!(location_buf),
            score_buf: empty_output_buffer!(score_buf),
            category_option: Some((empty_output_buffer!(categories_buf), categories_filter)),
            num_classes: 1,
            num_coords: 4,
            num_key_points: 0,
            num_values_per_key_point: 2,
            box_coord_offset: 0,
            keypoint_coord_offset: 4,
            x_scale: 0.0,
            y_scale: 0.0,
            w_scale: 0.0,
            h_scale: 0.0,
            anchors: None,
            min_score_threshold: 0.0,
            flip_vertically: false,
            score_clipping_thresh: None,
            apply_exponential_on_box_size: false,
            sigmoid_score: false,
        }
    }

    #[inline(always)]
    pub(crate) fn set_num_coords(&mut self, num_coords: usize) {
        self.num_coords = num_coords;
        check_valid!(self);
    }

    #[inline(always)]
    pub(crate) fn set_key_points(
        &mut self,
        num_key_points: usize,
        num_values_per_key_point: usize,
        keypoint_coord_offset: usize,
    ) {
        self.num_key_points = num_key_points;
        self.num_values_per_key_point = num_values_per_key_point;
        self.keypoint_coord_offset = keypoint_coord_offset;
        check_valid!(self);
    }

    #[inline(always)]
    pub(crate) fn set_anchors_scales(
        &mut self,
        x_scale: f32,
        y_scale: f32,
        w_scale: f32,
        h_scale: f32,
    ) {
        self.x_scale = x_scale;
        self.y_scale = y_scale;
        self.w_scale = w_scale;
        self.h_scale = h_scale;
    }

    #[inline(always)]
    pub(crate) fn set_sigmoid_score(&mut self, sigmoid_score: bool) {
        self.sigmoid_score = sigmoid_score;
    }

    #[inline(always)]
    pub(crate) fn set_box_format(&mut self, box_format: DetectionBoxFormat) {
        self.box_format = box_format;
    }

    #[inline(always)]
    pub(crate) fn set_score_clipping_thresh(&mut self, score_clipping_thresh: f32) {
        self.score_clipping_thresh = Some(score_clipping_thresh);
    }

    #[inline(always)]
    pub(crate) fn set_nms_overlap_type(&mut self, overlap_type: NonMaxSuppressionOverlapType) {
        self.nms.set_overlap_type(overlap_type);
    }

    #[inline(always)]
    pub(crate) fn set_nms_algorithm(&mut self, algorithm: NonMaxSuppressionAlgorithm) {
        self.nms.set_algorithm(algorithm);
    }

    #[inline(always)]
    pub(crate) fn set_nms_min_suppression_threshold(&mut self, min_suppression_threshold: f32) {
        self.nms
            .set_min_suppression_threshold(min_suppression_threshold);
    }

    #[inline(always)]
    pub(crate) fn location_buf(&mut self) -> &mut [u8] {
        self.location_buf.data_buffer.as_mut_slice()
    }

    #[inline(always)]
    pub(crate) fn categories_buf(&mut self) -> Option<&mut [u8]> {
        if let Some((ref mut c, _)) = self.category_option {
            return Some(c.data_buffer.as_mut_slice());
        }
        None
    }

    #[inline(always)]
    pub(crate) fn score_buf(&mut self) -> &mut [u8] {
        self.score_buf.data_buffer.as_mut_slice()
    }

    #[inline(always)]
    pub(crate) fn realloc(&mut self, num_boxes: usize) {
        realloc_output_buffer!(self.score_buf, num_boxes * self.num_classes);
        if let Some(ref mut c) = self.category_option {
            realloc_output_buffer!(c.0, num_boxes);
        }
        realloc_output_buffer!(self.location_buf, num_boxes * num_boxes);
    }

    pub(crate) fn result(&mut self, num_boxes: usize) -> DetectionResult {
        let scores = output_buffer_mut_slice!(self.score_buf);
        let location = output_buffer_mut_slice!(self.location_buf);
        let category_option = if let Some(ref mut c) = self.category_option {
            assert_eq!(self.num_classes, 1);
            Some((output_buffer_mut_slice!(c.0), &c.1))
        } else {
            None
        };

        // check buf if is valid
        debug_assert!(location.len() >= num_boxes * self.num_coords);
        debug_assert!(scores.len() >= num_boxes * self.num_classes);
        if let Some(a) = self.anchors {
            debug_assert_eq!(a.len(), num_boxes);
        }

        let mut detections = Vec::with_capacity(num_boxes);
        if let Some((categories_buf, categories_filter)) = category_option {
            let mut index = 0;
            for i in 0..num_boxes {
                if let Some(category) =
                    categories_filter.create_category(categories_buf[i] as usize, scores[i])
                {
                    if let Some(d) =
                        Self::generate_detection(&self.box_indices, category, location, index)
                    {
                        detections.push(d);
                    }
                }
                index += self.num_coords;
            }
        } else {
            let anchors = self.anchors.unwrap();
            let mut index = 0;
            let mut score_index = 0;
            for i in 0..num_boxes {
                let score = if self.num_classes == 1 {
                    scores[score_index]
                } else {
                    scores[score_index]
                };

                if score >= self.min_score_threshold {
                    let category = Category {
                        index: 0,
                        score,
                        category_name: None,
                        display_name: None,
                    };

                    Self::decode_boxes(
                        location,
                        anchors.as_slice(),
                        self.box_format,
                        num_boxes,
                        self.num_key_points,
                    );
                    if let Some(d) =
                        Self::generate_detection(&self.box_indices, category, location, index)
                    {
                        detections.push(d);
                    }
                }

                index += self.num_coords;
            }
        }

        let mut result = DetectionResult { detections };
        self.nms.do_nms(&mut result);
        result
    }

    #[inline(always)]
    fn generate_detection(
        box_indices: &[usize],
        category: Category,
        location: &[f32],
        index: usize,
    ) -> Option<Detection> {
        let rect = Rect {
            left: box_x_min!(box_indices, location, index),
            top: box_y_min!(box_indices, location, index),
            right: box_x_max!(box_indices, location, index),
            bottom: box_y_max!(box_indices, location, index),
        };
        if rect.left.is_nan()
            || rect.right.is_nan()
            || rect.top.is_nan()
            || rect.bottom.is_nan()
            || rect.left >= rect.right
            || rect.top >= rect.bottom
        {
            return None;
        }
        // todo: keypoint

        Some(Detection {
            categories: vec![category],
            bounding_box: rect,
            key_points: None,
        })
    }

    fn decode_boxes(
        raw_boxes: &mut [f32],
        anchors: &[Anchor],
        box_format: DetectionBoxFormat,
        num_boxes: usize,
        key_point_num: usize,
    ) {
        let mut index = 0;
        for i in 0..num_boxes {
            let mut x_center;
            let mut y_center;
            let h;
            let w;
            match box_format {
                DetectionBoxFormat::YXHW => {
                    y_center = raw_boxes[0];
                    x_center = raw_boxes[1];
                    h = raw_boxes[2];
                    w = raw_boxes[3];
                }
                DetectionBoxFormat::XYWH => {
                    x_center = raw_boxes[0];
                    y_center = raw_boxes[1];
                    w = raw_boxes[2];
                    h = raw_boxes[3];
                }
                DetectionBoxFormat::XYXY => {
                    x_center = (-raw_boxes[0] + raw_boxes[2]) / 2.;
                    y_center = (-raw_boxes[0 + 1] + raw_boxes[3]) / 2.;
                    w = raw_boxes[2] + raw_boxes[0];
                    h = raw_boxes[3] + raw_boxes[1];
                }
            }

            // todo: x_scale, y_scale, h_scale, w_scale
            let anchor = &anchors[i];
            x_center = x_center / anchor.w + anchor.x_center;
            y_center = y_center / anchor.h + anchor.y_center;

            let h_div_2 = h / 2.;
            let w_div_2 = w / 2.;
            let ymin = y_center - h_div_2;
            let xmin = x_center - w_div_2;
            let ymax = y_center + h_div_2;
            let xmax = x_center + w_div_2;
            raw_boxes[index] = ymin;
            raw_boxes[index + 1] = xmin;
            raw_boxes[index + 2] = ymax;
            raw_boxes[index + 3] = xmax;

            // todo: key points
            index += 4 + (key_point_num << 1);
        }
    }
}
