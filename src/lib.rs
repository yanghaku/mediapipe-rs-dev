mod error;
#[macro_use]
mod model_resource;

pub mod postprocess;
pub mod preprocess;
pub mod tasks;

pub use error::Error;
pub use model_resource::ModelResourceTrait;
pub use wasi_nn_safe::GraphExecutionTarget as Device;
use wasi_nn_safe::{
    Graph, GraphBuilder, GraphEncoding, GraphExecutionContext, SharedSlice, TensorType,
};
