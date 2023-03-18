use mediapipe_rs::tasks::vision::ImageClassifierBuilder;

const MODEL_1: &'static str = "assets/models/image_classification/efficientnet_lite0_fp32.tflite";
const MODEL_2: &'static str = "assets/models/image_classification/efficientnet_lite0_uint8.tflite";
const MODEL_3: &'static str = "assets/models/image_classification/efficientnet_lite2_fp32.tflite";
const MODEL_4: &'static str = "assets/models/image_classification/efficientnet_lite2_uint8.tflite";
const IMG_1: &'static str = "assets/testdata/img/cat.png";
const IMG_2: &'static str = "assets/testdata/img/banana.jpg";

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
        .max_results(5)
        .finalize()
        .unwrap();

    let mut session = image_classifier.new_session().unwrap();
    let res_1 = session.classify(&image::open(IMG_1).unwrap()).unwrap();
    println!("{}", res_1);
    let res_2 = session.classify(&image::open(IMG_2).unwrap()).unwrap();
    println!("{}", res_2);
}
