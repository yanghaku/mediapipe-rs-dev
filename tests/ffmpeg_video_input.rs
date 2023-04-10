#[cfg(feature = "ffmpeg")]
mod ffmpeg {
    use mediapipe_rs::preprocess::vision::FFMpegVideoData;
    use mediapipe_rs::tasks::vision::{ImageClassifierBuilder, ObjectDetectorBuilder};

    const IMAGE_CLASSIFICATION_MODEL: &'static str =
        "assets/models/image_classification/efficientnet_lite0_fp32.tflite";
    const OBJECT_DETECTION_MODEL: &'static str =
        "assets/models/object_detection/efficientdet_lite0_fp32.tflite";

    const VIDEO_1: &'static str = "assets/testdata/video/banana_clock_tabby.mp4";

    #[test]
    fn test_image_classification() {
        ffmpeg_next::init().unwrap();
        let input = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();

        let classification_results = ImageClassifierBuilder::new()
            .model_asset_path(IMAGE_CLASSIFICATION_MODEL)
            .max_results(1)
            .finalize()
            .unwrap()
            .classify_for_video(input)
            .unwrap();
        assert_eq!(classification_results.len(), 3);
        assert_eq!(
            classification_results[0].classifications[0].categories[0].category_name,
            Some("banana".into())
        );
        assert_eq!(
            classification_results[1].classifications[0].categories[0].category_name,
            Some("analog clock".into())
        );
        assert_eq!(
            classification_results[2].classifications[0].categories[0].category_name,
            Some("tabby".into())
        );
    }

    #[test]
    fn test_object_detection() {
        ffmpeg_next::init().unwrap();
        let input = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();

        let detection_results = ObjectDetectorBuilder::new()
            .model_asset_path(OBJECT_DETECTION_MODEL)
            .max_results(1)
            .finalize()
            .unwrap()
            .detect_for_video(input)
            .unwrap();
        assert_eq!(detection_results.len(), 3);
        assert_eq!(
            detection_results[0].detections[0].categories[0].category_name,
            Some("banana".into())
        );
        assert_eq!(
            detection_results[1].detections[0].categories[0].category_name,
            Some("clock".into())
        );
        assert_eq!(
            detection_results[2].detections[0].categories[0].category_name,
            Some("cat".into())
        );
        for result in detection_results {
            eprintln!("{}", result);
        }
    }

    #[test]
    fn test_results_iter() {
        ffmpeg_next::init().unwrap();
        let input_1 = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();
        let input_2 = FFMpegVideoData::new(ffmpeg_next::format::input(&VIDEO_1).unwrap()).unwrap();

        let classifier = ImageClassifierBuilder::new()
            .model_asset_path(IMAGE_CLASSIFICATION_MODEL)
            .max_results(1)
            .finalize()
            .unwrap();
        let mut session = classifier.new_session().unwrap();

        let mut results_iter = session.classify_for_video(input_1).unwrap();
        let mut num_frame = 0;
        while let Some(result) = results_iter.next().unwrap() {
            eprintln!("Frame {}: {}", num_frame, result);
            num_frame += 1;
        }

        let mut results_iter = session.classify_for_video(input_2).unwrap();
        num_frame = 0;
        while let Some(result) = results_iter.next().unwrap() {
            eprintln!("Frame {}: {}", num_frame, result);
            num_frame += 1;
        }
    }
}
