use crate::postprocess::DetectionResult;
use image::GenericImage;

pub fn draw_detection(img: &mut image::DynamicImage, det: &DetectionResult) {
    let border_pixel = image::Rgba::from([255u8, 0u8, 0u8, 1u8]);

    for d in &det.detections {
        let x_min = (d.bounding_box.left * img.width() as f32) as u32;
        let x_max = (d.bounding_box.right * img.width() as f32) as u32;
        let y_min = (d.bounding_box.top * img.height() as f32) as u32;
        let y_max = (d.bounding_box.bottom * img.height() as f32) as u32;
        for x in x_min..=x_max {
            img.put_pixel(x, y_min, border_pixel.clone());
            img.put_pixel(x, y_max, border_pixel.clone());
        }
        for y in y_min..=y_max {
            img.put_pixel(x_min, y, border_pixel.clone());
            img.put_pixel(x_max, y, border_pixel.clone());
        }
    }
}
