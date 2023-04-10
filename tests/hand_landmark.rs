use mediapipe_rs::tasks::vision::HandLandmarkerBuilder;

const MODEL_PATH: &'static str = "assets/models/hand_landmark_detection/hand_landmarker.task";
const HANDS_1: &'static str = "assets/testdata/img/google_sample_woman_hands.jpg";

#[test]
fn test_hand_detection() {
    let hand_landmarker = HandLandmarkerBuilder::new()
        .model_asset_path(MODEL_PATH)
        .finalize()
        .unwrap();
    let hand_detector = hand_landmarker.subtask();
    let img = image::open(HANDS_1).unwrap();
    let detection_result = hand_detector.detect(&img).unwrap();
    assert_eq!(detection_result.detections.len(), 2);
    eprintln!("{}", detection_result);
}

#[test]
fn test_hand_landmark() {
    let hand_landmark_results = HandLandmarkerBuilder::new()
        .model_asset_path(MODEL_PATH)
        .cpu()
        .finalize()
        .unwrap();
}
