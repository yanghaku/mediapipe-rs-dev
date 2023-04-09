use crate::model::MemoryTextFile;
use crate::postprocess::Category;
use crate::tasks::common::ClassificationOptions;
use std::borrow::Cow;
use std::collections::HashSet;

enum Label<'a> {
    Deny,
    Allowed((Cow<'a, str>, Option<Cow<'a, str>>)),
}

pub(crate) struct CategoriesFilter<'a> {
    labels: Vec<Label<'a>>,
    score_threshold: f32,
}

impl<'a> CategoriesFilter<'a> {
    /// create a empty categories filter, just filter the score threshold.
    #[inline(always)]
    pub(crate) fn new_empty(score_threshold: f32) -> Self {
        Self {
            labels: Vec::new(),
            score_threshold,
        }
    }

    pub(crate) fn new(
        option: &ClassificationOptions,
        labels: &'a [u8],
        labels_locale: Option<&'a [u8]>,
    ) -> Self {
        let mut is_allow_list = false;
        let mut set = HashSet::new();
        if !option.category_deny_list.is_empty() {
            set.extend(option.category_deny_list.iter().map(|s| s.as_str()));
        }
        if !option.category_allow_list.is_empty() {
            is_allow_list = true;
            set.extend(option.category_allow_list.iter().map(|s| s.as_str()));
        }

        let mut label_file = MemoryTextFile::new(labels);
        let mut vec = Vec::with_capacity(set.len());
        while let Some(line) = label_file.next_line() {
            let allow = if set.contains(line.as_ref()) {
                is_allow_list
            } else {
                !is_allow_list
            };

            if allow {
                vec.push(Label::Allowed((line, None)));
            } else {
                vec.push(Label::Deny);
            }
        }

        if let Some(labels_locale) = labels_locale {
            let mut iter = vec.iter_mut();
            let mut labels_locale_file = MemoryTextFile::new(labels_locale);
            while let Some(line) = labels_locale_file.next_line() {
                if let Some(Label::Allowed((_, o))) = iter.next() {
                    *o = Some(line);
                }
            }
        }

        Self {
            labels: vec,
            score_threshold: option.score_threshold,
        }
    }

    #[inline(always)]
    pub fn create_category(&self, index: usize, score: f32) -> Option<Category> {
        if score >= self.score_threshold {
            if let Some(Label::Allowed((l, l_locale))) = self.labels.get(index) {
                return Some(Category {
                    index: index as u32,
                    score,
                    category_name: Some(l.clone().into_owned()),
                    display_name: l_locale.clone().map(|l| l.into_owned()),
                });
            }
        }
        None
    }
}
