use mediapipe_rs::postprocess::ImageCategoryMask;
use mediapipe_rs::tasks::vision::ImageSegmenterBuilder;

const MODEL_1: &'static str = "assets/models/image_segmentation/deeplabv3.tflite";
const MODEL_2: &'static str = "assets/models/image_segmentation/selfie_segm_128_128_3.tflite";
const IMG_1: &'static str = "/assets/testdata/img/cat_and_dog.jpg";

#[test]
fn test_image_segmentation_model_1() {
    test_image_segmentation_tasks(MODEL_1)
}

#[test]
fn test_image_segmentation_model_2() {
    test_image_segmentation_tasks(MODEL_2)
}

fn test_image_segmentation_tasks(model_asset: &str) {
    let img = image::open(IMG_1).unwrap();
    let segmentation_res = ImageSegmenterBuilder::new()
        .model_asset_path(model_asset)
        .output_confidence_masks(true)
        .output_category_mask(true)
        .finalize()
        .unwrap()
        .segment(&img)
        .unwrap();
    assert!(segmentation_res.confidence_masks.is_some());
    assert!(segmentation_res.category_mask.is_some());
    let confidence_masks = segmentation_res.confidence_masks.as_ref().unwrap();
    let category_mask = segmentation_res.category_mask.as_ref().unwrap();
    for c in confidence_masks {
        assert_eq!(c.width(), img.width());
        assert_eq!(c.height(), img.height());
    }
    assert_eq!(category_mask.width(), img.width());
    assert_eq!(category_mask.height(), img.height());

    let draw = false;
    if draw {
        let path = std::path::PathBuf::from(model_asset);
        let out_file = path.file_stem().to_owned().unwrap().to_str().unwrap();
        draw_mask(
            img.to_rgb8(),
            category_mask,
            format!("./target/{}.jpg", out_file).as_str(),
        );
    }
}

#[allow(unused)]
fn draw_mask(img: image::RgbImage, mask: &ImageCategoryMask, path: &str) {
    let mut out_img = image::imageops::blur(&img, 10.);
    for x in 0..mask.width() {
        for y in 0..mask.height() {
            if mask.get_pixel(x, y).0[0] > 0 {
                out_img.put_pixel(x, y, img.get_pixel(x, y).clone());
            }
        }
    }
    out_img.save(path).unwrap();
}
