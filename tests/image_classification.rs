use mediapipe_rs::tasks::vision::ImageClassifierBuilder;

const MODEL_1: &'static str = "assets/models/image_classification/efficientnet_lite0_fp32.tflite";
const MODEL_2: &'static str = "assets/models/image_classification/efficientnet_lite0_uint8.tflite";
const MODEL_3: &'static str = "assets/models/image_classification/efficientnet_lite2_fp32.tflite";
const MODEL_4: &'static str = "assets/models/image_classification/efficientnet_lite2_uint8.tflite";
const IMG: &'static str = "assets/testdata/img/banana.jpg";

#[test]
fn test_image_classification_model_1() {
    image_classification_task_run(MODEL_1.to_string());
}

#[test]
fn test_image_classification_model_2() {
    image_classification_task_run(MODEL_2.to_string());
}

#[test]
fn test_image_classification_model_3() {
    image_classification_task_run(MODEL_3.to_string());
}

#[test]
fn test_image_classification_model_4() {
    image_classification_task_run(MODEL_4.to_string());
}

fn image_classification_task_run(model_asset_path: String) {
    let image_classifier = ImageClassifierBuilder::new()
        .model_asset_path(model_asset_path)
        .cpu()
        .max_results(2)
        .finalize()
        .unwrap();

    let res = image_classifier
        .classify(&image::open(IMG).unwrap())
        .unwrap();
    eprintln!("{}", res);
    // banana: 954
    let top = res
        .classifications
        .get(0)
        .unwrap()
        .categories
        .get(0)
        .unwrap();
    assert_eq!(top.index, 954);
    assert_eq!(top.category_name.as_ref().unwrap().as_str(), "banana");
}

#[test]
fn test_bird_from_tf_hub() {
    const MODEL: &'static str =
        "assets/models/image_classification/lite-model_aiy_vision_classifier_birds_V1_3.tflite";
    const IMAGE: &'static str = "assets/testdata/img/bird.jpg";

    let res = ImageClassifierBuilder::new()
        .model_asset_path(MODEL.to_string())
        .max_results(2)
        .finalize()
        .unwrap()
        .classify(&image::open(IMAGE).unwrap())
        .unwrap();
    assert_eq!(
        res.classifications[0].categories[0]
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "/m/01bwb9"
    );
    assert_eq!(
        res.classifications[0].categories[0]
            .display_name
            .as_ref()
            .unwrap()
            .as_str(),
        "Passer domesticus"
    );
    eprintln!("{}", res);
}
