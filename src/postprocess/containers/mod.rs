// NOTE: The code in module containers is ported from c++ in google mediapipe [1].
// [1]: https://github.com/google/mediapipe/

pub(super) mod category;
pub(super) mod classification_result;

#[cfg(feature = "vision")]
pub(super) mod detection_result;
#[cfg(feature = "vision")]
pub(super) mod key_point;
#[cfg(feature = "vision")]
pub(super) mod rect;
