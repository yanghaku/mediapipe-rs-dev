//!
//! # A Rust library for mediapipe tasks for WasmEdge WASI-NN
//!
//! ## Introduction
//!
//! * **Easy to use**: low-code APIs such as mediapipe-python.
//! * **Low overhead**: No unnecessary data copy, allocation, and free during the processing.
//! * **Flexible**: Users can use custom media bytes as input, such as ```ndarray```.
//! * For TfLite models, the library not only supports all models downloaded from [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/)
//!   but also supports **[TF Hub](https://tfhub.dev/)** models and **custom models** with essential information.
//! * Support **multiple model formats**, such as TfLite, PyTorch, and Onnx.
//!   The library can **detect it when loading models**.
//!
//! ## Examples
//!
//! ### Image classification
//!
//! ```rust
//! use mediapipe_rs::tasks::vision::ImageClassifierBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let (model_path, img_path) = parse_args()?;
//!
//!     let classification_result = ImageClassifierBuilder::new()
//!         .model_asset_path(model_path) // set model path
//!         .max_results(4) // set max result
//!         .finalize()? // create a image classifier
//!         .classify(&image::open(img_path)?)?; // do inference and generate results
//!
//!     //! show formatted result message
//!     println!("{}", classification_result);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Object Detection
//!
//! ```rust
//! use mediapipe_rs::postprocess::utils::draw_detection;
//! use mediapipe_rs::tasks::vision::ObjectDetectorBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let (model_path, img_path, output_path) = parse_args()?;
//!
//!     let mut input_img = image::open(img_path)?;
//!     let detection_result = ObjectDetectorBuilder::new()
//!         .model_asset_path(model_path) // set model path
//!         .max_results(2) // set max result
//!         .finalize()? // create a object detector
//!         .detect(&input_img)?; // do inference and generate results
//!
//!     // show formatted result message
//!     println!("{}", detection_result);
//!
//!     if let Some(output_path) = output_path {
//!         // draw detection result to image
//!         draw_detection(&mut input_img, &detection_result);
//!         // save output image
//!         input_img.save(output_path)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Text Classification
//! ```rust
//! use mediapipe_rs::tasks::text::TextClassifierBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model_path = parse_args()?;
//!
//!     let text_classifier = TextClassifierBuilder::new()
//!         .model_asset_path(model_path) // set model path
//!         .max_results(1) // set max result
//!         .finalize()?; // create a text classifier
//!
//!     let positive_str = "I love coding so much!";
//!     let negative_str = "I don't like raining.";
//!
//!     // classify show formatted result message
//!     let result = text_classifier.classify(&positive_str)?;
//!     println!("`{}` -- {}", positive_str, result);
//!
//!     let result = text_classifier.classify(&negative_str)?;
//!     println!("`{}` -- {}", negative_str, result);
//!
//!     Ok(())
//! }
//! ```
//! ### Audio Input
//!
//! Every media which implements the trait [`preprocess::audio::AudioData`] or trait [`preprocess::InToTensorsIterator`], can be used as audio tasks input.
//! Now the library has builtin implementation to support ```symphonia```, ```ffmpeg```, and raw audio data as input.
//!
//! Examples for Audio Classification:
//!
//! ```rust
//! use mediapipe_rs::tasks::audio::AudioClassifierBuilder;
//!
//! #[cfg(feature = "ffmpeg")]
//! use mediapipe_rs::preprocess::audio::FFMpegAudioData;
//! #[cfg(not(feature = "ffmpeg"))]
//! use mediapipe_rs::preprocess::audio::SymphoniaAudioData;
//!
//! #[cfg(not(feature = "ffmpeg"))]
//! fn read_audio_using_symphonia(audio_path: String) -> SymphoniaAudioData {
//!     let file = std::fs::File::open(audio_path).unwrap();
//!     let probed = symphonia::default::get_probe()
//!         .format(
//!             &Default::default(),
//!             symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default()),
//!             &Default::default(),
//!             &Default::default(),
//!         )
//!         .unwrap();
//!     let codec_params = &probed.format.default_track().unwrap().codec_params;
//!     let decoder = symphonia::default::get_codecs()
//!         .make(codec_params, &Default::default())
//!         .unwrap();
//!     SymphoniaAudioData::new(probed.format, decoder)
//! }
//!
//! #[cfg(feature = "ffmpeg")]
//! fn read_video_using_ffmpeg(audio_path: String) -> FFMpegAudioData {
//!     ffmpeg_next::init().unwrap();
//!     FFMpegAudioData::new(ffmpeg_next::format::input(&audio_path.as_str()).unwrap()).unwrap()
//! }
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let (model_path, audio_path) = parse_args()?;
//!
//!     #[cfg(not(feature = "ffmpeg"))]
//!     let audio = read_audio_using_symphonia(audio_path);
//!     #[cfg(feature = "ffmpeg")]
//!     let audio = read_video_using_ffmpeg(audio_path);
//!
//!     let classification_results = AudioClassifierBuilder::new()
//!         .model_asset_path(model_path) // set model path
//!         .max_results(3) // set max result
//!         .finalize()? // create a image classifier
//!         .classify(audio)?; // do inference and generate results
//!
//!     // show formatted result message
//!     for c in classification_results {
//!         println!("{}", c);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Use the Session to speed up
//!
//! The session includes inference sessions (such as TfLite interpreter), input and output buffers, etc.
//! Explicitly using the session can reuse these resources to speed up.
//!
//! ### Example: Text Classificaton
//!
//! Origin:
//! ```rust
//! use mediapipe_rs::tasks::text::TextClassifier;
//! use mediapipe_rs::postprocess::ClassificationResult;
//! use mediapipe_rs::Error;
//!
//! fn inference(
//!     text_classifier: &TextClassifier,
//!     inputs: &Vec<String>
//! ) -> Result<Vec<ClassificationResult>, Error> {
//!     let mut res = Vec::with_capacity(inputs.len());
//!     for input in inputs {
//!         // text_classifier will create new session every time
//!         res.push(text_classifier.classify(input.as_str())?);
//!     }
//!     Ok(res)
//! }
//! ```
//!
//! Use the session to speed up:
//! ```rust
//! use mediapipe_rs::tasks::text::TextClassifier;
//! use mediapipe_rs::postprocess::ClassificationResult;
//! use mediapipe_rs::Error;
//!
//! fn inference(
//!     text_classifier: &TextClassifier,
//!     inputs: &Vec<String>
//! ) -> Result<Vec<ClassificationResult>, Error> {
//!     let mut res = Vec::with_capacity(inputs.len());
//!     // only create one session and reuse the resources in session.
//!     let mut session = text_classifier.new_session()?;
//!     for input in inputs {
//!         res.push(session.classify(input.as_str())?);
//!     }
//!     Ok(res)
//! }
//! ```
//!
//! ## GPU and TPU support
//!
//! The default device is CPU, and user can use APIs to choose device to use:
//! ```rust
//! use mediapipe_rs::tasks::vision::ObjectDetectorBuilder;
//!
//! fn create_gpu(model_blob: Vec<u8>) {
//!     let detector_gpu = ObjectDetectorBuilder::new()
//!         .model_asset_buffer(model_blob)
//!         .gpu()
//!         .finalize()
//!         .unwrap();
//! }
//!
//! fn create_tpu(model_blob: Vec<u8>) {
//!     let detector_tpu = ObjectDetectorBuilder::new()
//!         .model_asset_buffer(model_blob)
//!         .tpu()
//!         .finalize()
//!         .unwrap();
//! }
//! ```

#[cfg(not(any(feature = "vision", feature = "audio", feature = "text")))]
compile_error!("Must select at least one task type: `vision`, `audio`, `text`");

mod error;
#[macro_use]
mod model;

pub mod postprocess;
pub mod preprocess;
pub mod tasks;

pub use error::Error;
pub use wasi_nn_safe::GraphExecutionTarget as Device;
use wasi_nn_safe::{
    Graph, GraphBuilder, GraphEncoding, GraphExecutionContext, SharedSlice, TensorType,
};
