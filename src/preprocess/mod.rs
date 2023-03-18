pub mod audio;

pub mod text;

pub mod vision;

use crate::model_resource::ModelResourceTrait;
use crate::Error;

/// Every media such as Image, Audio, Text, can implement this trait and be used as model input
pub trait ToTensor<'t> {
    type OutputType: AsRef<[u8]> + 't;

    fn to_tensor(
        &self,
        input_index: usize,
        model_resource: &Box<dyn ModelResourceTrait>,
    ) -> Result<Self::OutputType, Error>;
}
