/// result containers
mod containers;
pub use containers::*;

/// stateless operators for tensor
mod ops;
pub use ops::{Activation, QuantizationParameters};

/// stateful objects, convert tensor to results
mod processing;
pub(crate) use processing::*;

/// utils to use the results, such as draw_utils
pub mod utils;
