use super::*;
use crate::tasks::common::ClassifierBuilder;

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
    ($self:ident, $v:ident, $offset:ident) => {
        $v[$offset + $self.box_indices[0]]
    };
}

macro_rules! box_x_min {
    ($self:ident, $v:ident, $offset:ident) => {
        $v[$offset + $self.box_indices[1]]
    };
}

macro_rules! box_y_max {
    ($self:ident, $v:ident, $offset:ident) => {
        $v[$offset + $self.box_indices[2]]
    };
}

macro_rules! box_x_max {
    ($self:ident, $v:ident, $offset:ident) => {
        $v[$offset + $self.box_indices[3]]
    };
}

pub(crate) struct DetectionSession<'a> {
    options: &'a ClassifierBuilder,
    anchors: Option<Vec<Anchor>>,
    box_indices: [usize; 4],
    box_format: DetectionBoxFormat,
    nms: NonMaxSuppressionBuilder,

    location_buf: OutputBuffer,
    categories_buf: OutputBuffer,
    score_buf: OutputBuffer,
}

impl<'a> DetectionSession<'a> {
    #[inline]
    pub(crate) fn new(
        options: &'a ClassifierBuilder,
        bound_box_properties: &'a [usize; 4],
        location_buf: (TensorType, Option<QuantizationParameters>),
        categories_buf: (TensorType, Option<QuantizationParameters>),
        score_buf: (TensorType, Option<QuantizationParameters>),
    ) -> Self {
        Self {
            options,
            anchors: None,
            box_format: Default::default(),
            box_indices: [
                bound_box_properties[1], // y_min
                bound_box_properties[0], // x_min
                bound_box_properties[3], // y_max
                bound_box_properties[2], // x_max
            ],
            nms: NonMaxSuppressionBuilder::new(options.max_results, options.score_threshold),
            location_buf: empty_output_buffer!(location_buf),
            categories_buf: empty_output_buffer!(categories_buf),
            score_buf: empty_output_buffer!(score_buf),
        }
    }

    pub(crate) fn location_buf(&mut self) -> &mut [u8] {
        self.location_buf.data_buffer.as_mut_slice()
    }

    pub(crate) fn categories_buf(&mut self) -> &mut [u8] {
        self.categories_buf.data_buffer.as_mut_slice()
    }

    pub(crate) fn score_buf(&mut self) -> &mut [u8] {
        self.score_buf.data_buffer.as_mut_slice()
    }

    pub(crate) fn realloc(&mut self, num_boxes: usize, key_point_num: usize) {
        realloc_output_buffer!(self.score_buf, num_boxes);
        realloc_output_buffer!(self.categories_buf, num_boxes);
        let location_f32_num = (num_boxes << 2) + (key_point_num << 1);
        realloc_output_buffer!(self.location_buf, location_f32_num);
    }

    pub(crate) fn result(&mut self, num_boxes: usize, key_point_num: usize) -> DetectionResult {
        let scores = output_buffer_mut_slice!(self.score_buf);
        let location = output_buffer_mut_slice!(self.location_buf);
        let category = output_buffer_mut_slice!(self.categories_buf);

        if let Some(anchors) = self.anchors.as_ref() {
            Self::decode_boxes(
                location,
                anchors.as_slice(),
                self.box_format,
                num_boxes,
                key_point_num,
            );
        }

        let mut detections = Vec::with_capacity(num_boxes);
        let mut index = 0;
        let score_threshold = self.options.score_threshold;

        for i in 0..num_boxes {
            let score = scores[i];
            // todo: allow_list
            if score < score_threshold {
                index += 4 + (key_point_num << 1);
                continue;
            }

            let rect = Rect {
                left: box_x_min!(self, location, index),
                top: box_y_min!(self, location, index),
                right: box_x_max!(self, location, index),
                bottom: box_y_max!(self, location, index),
            };

            index += 4;
            // todo: keypoint
            index += key_point_num << 1;

            detections.push(Detection {
                categories: vec![Category {
                    index: category[i] as i32,
                    score: scores[i],
                    category_name: None,
                    display_name: None,
                }],
                bounding_box: rect,
                key_points: None,
            });
        }

        let mut result = DetectionResult { detections };
        self.nms.do_nms(&mut result);
        result
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
