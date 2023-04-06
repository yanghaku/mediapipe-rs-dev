#[macro_use]
pub(crate) mod common;

#[cfg(feature = "audio")]
pub mod audio;

#[cfg(feature = "text")]
pub mod text;

#[cfg(feature = "vision")]
pub mod vision;

/// Task session trait to process the stream data
pub trait TaskSession {
    type Result: 'static;

    /// process the next tensors from input stream
    fn process_next<TensorsIter: crate::preprocess::TensorsIterator>(
        &mut self,
        input_tensors_iter: &mut TensorsIter,
    ) -> Result<Option<Self::Result>, crate::Error>;
}
