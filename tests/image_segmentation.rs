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
}
