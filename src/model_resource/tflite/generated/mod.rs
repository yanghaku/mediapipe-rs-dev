#![allow(non_snake_case)]
#![allow(unused_imports)]

mod metadata_schema_generated;
mod schema_generated;

pub(crate) use metadata_schema_generated::tflite as tflite_metadata;
pub(crate) use schema_generated::tflite;
