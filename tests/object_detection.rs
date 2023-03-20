use mediapipe_rs::tasks::vision::ObjectDetectorBuilder;

const MODEL_1: &'static str = "assets/models/object_detection/efficientdet_lite0_fp32.tflite";
const MODEL_2: &'static str = "assets/models/object_detection/efficientdet_lite0_uint8.tflite";
const MODEL_3: &'static str = "assets/models/object_detection/efficientdet_lite2_fp32.tflite";
const MODEL_4: &'static str = "assets/models/object_detection/efficientdet_lite2_uint8.tflite";
const MODEL_5: &'static str = "assets/models/object_detection/mobilenetv2_ssd_256_fp32.tflite";
const MODEL_6: &'static str = "assets/models/object_detection/mobilenetv2_ssd_256_uint8.tflite";
const IMG: &'static str = "assets/testdata/img/cat_and_dog.jpg";

#[test]
fn test_object_detection_model_1() {
    object_detection_task_run(MODEL_1.to_string());
}

#[test]
fn test_object_detection_model_2() {
    object_detection_task_run(MODEL_2.to_string());
}

#[test]
fn test_object_detection_model_3() {
    object_detection_task_run(MODEL_3.to_string());
}

#[test]
fn test_object_detection_model_4() {
    object_detection_task_run(MODEL_4.to_string());
}

#[test]
fn test_object_detection_model_5() {
    object_detection_task_run(MODEL_5.to_string());
}

#[test]
fn test_object_detection_model_6() {
    object_detection_task_run(MODEL_6.to_string());
}

fn object_detection_task_run(model_asset_path: String) {
    let object_detector = ObjectDetectorBuilder::new()
        .model_asset_path(model_asset_path)
        .cpu()
        .max_results(5)
        .finalize()
        .unwrap();

    let img = image::open(IMG).unwrap();
    let mut session = object_detector.new_session().unwrap();
    let res = session.detect(&img).unwrap();
    eprintln!("{}", res);
}
