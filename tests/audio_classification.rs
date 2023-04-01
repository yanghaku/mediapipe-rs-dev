use mediapipe_rs::preprocess::audio::AudioData;
use mediapipe_rs::tasks::audio::AudioClassifierBuilder;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::io::MediaSourceStream;

fn read_audio(audio_path: impl AsRef<std::path::Path>) -> AudioData {
    let file = std::fs::File::open(audio_path).unwrap();
    let mut probed = symphonia::default::get_probe()
        .format(
            &Default::default(),
            MediaSourceStream::new(Box::new(file), Default::default()),
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
    let codec_params = &probed.format.default_track().unwrap().codec_params;
    let mut decoder = symphonia::default::get_codecs()
        .make(codec_params, &Default::default())
        .unwrap();

    let mut sample_rate = 0;
    let mut buf = Vec::new();
    while let Ok(p) = probed.format.next_packet() {
        match decoder.decode(&p).unwrap() {
            AudioBufferRef::S16(r) => {
                let max = i16::MAX as f32;
                let spec = r.spec();
                sample_rate = spec.rate;
                let channels = spec.channels.count();
                while buf.len() < channels {
                    buf.push(Vec::new())
                }
                for c in 0..channels {
                    let m = buf.get_mut(c).unwrap();
                    for s in r.chan(c) {
                        m.push(*s as f32 / max)
                    }
                }
            }
            _ => {
                unimplemented!()
            }
        }
    }

    assert_ne!(sample_rate, 0);
    AudioData::new(buf, sample_rate)
}

const MODEL_1: &'static str =
    "assets/models/audio_classification/yamnet_audio_classifier_with_metadata.tflite";

const AUDIO_PATH: &'static str = "assets/testdata/audio/speech_16000_hz_mono.wav";

#[test]
fn test_audio_classification() {
    audio_classification_task_run(MODEL_1.to_string());
}

fn audio_classification_task_run(model_asset_path: String) {
    let input = read_audio(AUDIO_PATH);

    let classification_list = AudioClassifierBuilder::new()
        .model_asset_path(model_asset_path)
        .cpu()
        .max_results(1)
        .finalize()
        .unwrap()
        .classify(&input)
        .unwrap();
    for classification in &classification_list {
        eprintln!("{}", classification);
    }

    assert_eq!(classification_list.len(), 5);
    assert_eq!(
        classification_list[0].classifications[0].categories[0].index,
        0
    );
}
