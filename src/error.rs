use wasi_nn_safe::{thiserror, Error as WasiNNError};

/// Mediapipe-rs API error enum.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Wasi-NN Error: {0}")]
    WasiNNError(#[from] WasiNNError),

    #[error("Argument Error: {0}")]
    ArgumentError(String),

    #[error("FlatBuffer Error: {0}")]
    FlatBufferError(#[from] flatbuffers::InvalidFlatbuffer),

    #[error("Model Binary Parse Error: {0}")]
    ModelParseError(String),

    #[error("Model Inconsistent Error: {0}")]
    ModelInconsistentError(String),
}
