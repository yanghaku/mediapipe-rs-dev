use mediapipe_rs::tasks::vision::HandLandmarkerBuilder;

const MODEL_PATH: &'static str = "assets/models/hand_landmark_detection/hand_landmarker.task";
const HANDS_1: &'static str = "assets/testdata/img/google_sample_woman_hands.jpg";

#[test]
fn test_hand_detection() {
    let hand_landmarker = HandLandmarkerBuilder::new()
        .model_asset_path(MODEL_PATH)
        .num_hands(5)
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
    let img = image::open(HANDS_1).unwrap();
    let hand_landmark_results = HandLandmarkerBuilder::new()
        .model_asset_path(MODEL_PATH)
        .cpu()
        .num_hands(10)
        .finalize()
        .unwrap()
        .detect(&img)
        .unwrap();
    assert_eq!(hand_landmark_results.len(), 2);
    assert_eq!(
        hand_landmark_results[0]
            .handedness
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "Right"
    );
    assert_eq!(
        hand_landmark_results[1]
            .handedness
            .category_name
            .as_ref()
            .unwrap()
            .as_str(),
        "Left"
    );
    eprintln!("{}", hand_landmark_results);

    let mut m = img.clone();
    let options = mediapipe_rs::postprocess::utils::DrawLandmarksOptions::default()
        .connections(mediapipe_rs::postprocess::HandLandmark::CONNECTIONS);
    for r in hand_landmark_results.iter() {
        mediapipe_rs::postprocess::utils::draw_landmarks_with_options(
            &mut m,
            &r.hand_landmarks,
            &options,
        );
    }
    m.save("./target/hand_landmark_test.jpg").unwrap();
}
