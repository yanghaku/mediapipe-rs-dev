mod bert_tensor;
mod regex_to_tensor;

use super::*;
use regex::Regex;
use std::borrow::Cow;
use std::collections::HashMap;

/// Necessary information for the text to tensor.
#[derive(Debug)]
pub enum TextToTensorInfo<'buf> {
    /// A BERT-based model.
    BertModel {
        token_index_map: HashMap<Cow<'buf, str>, i32>,

        /// maximum input sequence length for the bert and regex model.
        max_seq_len: u32,

        classifier_token_id: i32,
        separator_token_id: i32,
    },
    /// A model expecting input passed through a regex-based tokenizer.
    RegexModel {
        /// delim regex pattern
        delim_regex: Regex,

        token_index_map: HashMap<Cow<'buf, str>, i32>,

        /// maximum input sequence length for the bert and regex model.
        max_seq_len: u32,

        unknown_id: i32,
        pad_id: i32,
    },
    /// A model taking a string tensor input.
    StringModel,
    /// A UniversalSentenceEncoder-based model.
    UseModel,
}

macro_rules! check_map {
    ( $token_index_map:ident, $val:expr ) => {
        match $token_index_map.get($val) {
            Some(v) => v.clone(),
            None => {
                return Err(Error::ModelInconsistentError(format!(
                    "Vocabulary file doesn't have `{}` token.",
                    $val
                )));
            }
        }
    };
}

impl<'buf> TextToTensorInfo<'buf> {
    pub const REGEX_START_TOKEN: &'static str = "<START>";
    pub const REGEX_PAD_TOKEN: &'static str = "<PAD>";
    pub const REGEX_UNKNOWN_TOKEN: &'static str = "<UNKNOWN>";
    pub const BERT_CLASSIFIER_TOKEN: &'static str = "[CLS]";
    pub const BERT_SEPARATOR_TOKEN: &'static str = "[SEP]";

    pub fn new_regex_model(
        max_seq_len: u32,
        delim_regex_pattern: &str,
        token_index_map: HashMap<Cow<'buf, str>, i32>,
    ) -> Result<Self, Error> {
        // rust regex has no \'
        let delim_regex_pattern = delim_regex_pattern.replace(r"\'", r"'");
        let delim_regex =
            Regex::new(format!("({})", delim_regex_pattern).as_str()).map_err(|e| {
                Error::ModelInconsistentError(format!(
                    "Cannot parse delim regex pattern: `{:?}`",
                    e
                ))
            })?;
        let pad_id = check_map!(token_index_map, Self::REGEX_PAD_TOKEN);
        let unknown_id = check_map!(token_index_map, Self::REGEX_UNKNOWN_TOKEN);
        Ok(Self::RegexModel {
            delim_regex,
            token_index_map,
            max_seq_len,
            unknown_id,
            pad_id,
        })
    }

    pub fn new_bert_model(
        max_seq_len: u32,
        token_index_map: HashMap<Cow<'buf, str>, i32>,
    ) -> Result<Self, Error> {
        if max_seq_len < 2 {
            return Err(Error::ModelInconsistentError(
                "Bert model max seq length must be at least `2`".into(),
            ));
        }
        let classifier_token_id = check_map!(token_index_map, Self::BERT_CLASSIFIER_TOKEN);
        let separator_token_id = check_map!(token_index_map, Self::BERT_SEPARATOR_TOKEN);
        Ok(Self::BertModel {
            max_seq_len,
            token_index_map,
            classifier_token_id,
            separator_token_id,
        })
    }
}

impl ToTensor for &str {
    fn to_tensors(
        &self,
        _input_index: usize,
        model_resource: &Box<dyn ModelResourceTrait>,
        output_buffers: &mut [impl AsMut<[u8]>],
    ) -> Result<(), Error> {
        if let Some(info) = model_resource.text_to_tensor_info() {
            match info {
                TextToTensorInfo::BertModel {
                    max_seq_len,
                    token_index_map,
                    classifier_token_id,
                    separator_token_id,
                    ..
                } => {
                    debug_assert_eq!(output_buffers.len(), 3);
                    return bert_tensor::to_bert_tensors(
                        self,
                        token_index_map,
                        output_buffers,
                        *max_seq_len,
                        *classifier_token_id,
                        *separator_token_id,
                    );
                }
                TextToTensorInfo::RegexModel {
                    delim_regex,
                    token_index_map,
                    max_seq_len,
                    unknown_id,
                    pad_id,
                    ..
                } => {
                    debug_assert_eq!(output_buffers.len(), 1);
                    return regex_to_tensor::regex_to_tensors(
                        self,
                        delim_regex,
                        token_index_map,
                        &mut output_buffers[0],
                        *max_seq_len,
                        *unknown_id,
                        *pad_id,
                    );
                }
                TextToTensorInfo::StringModel | TextToTensorInfo::UseModel => {
                    todo!("Text String model")
                }
            }
        }
        Err(Error::ModelInconsistentError(format!(
            "Cannot get model text to tensor information."
        )))
    }
}

impl ToTensor for String {
    #[inline(always)]
    fn to_tensors(
        &self,
        input_index: usize,
        model_resource: &Box<dyn ModelResourceTrait>,
        output_buffers: &mut [impl AsMut<[u8]>],
    ) -> Result<(), Error> {
        self.as_str()
            .to_tensors(input_index, model_resource, output_buffers)
    }
}

impl<'a> ToTensor for Cow<'a, str> {
    #[inline(always)]
    fn to_tensors(
        &self,
        input_index: usize,
        model_resource: &Box<dyn ModelResourceTrait>,
        output_buffers: &mut [impl AsMut<[u8]>],
    ) -> Result<(), Error> {
        match self {
            Cow::Borrowed(s) => (*s).to_tensors(input_index, model_resource, output_buffers),
            Cow::Owned(s) => s.to_tensors(input_index, model_resource, output_buffers),
        }
    }
}
