use crate::postprocess::Category;
use crate::tasks::common::ClassifierBuilder;
use std::borrow::Cow;
use std::collections::HashSet;

enum Label<'a> {
    Deny,
    Allowed((String, Option<String>)),
    AllowedRef((&'a str, Option<&'a str>)),
}

pub(crate) struct CategoriesFilter<'a> {
    labels: Vec<Label<'a>>,
    score_threshold: f32,
}

impl<'a> CategoriesFilter<'a> {
    pub(crate) fn new(
        option: &ClassifierBuilder,
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

        let labels: Cow<'a, str> = String::from_utf8_lossy(labels);
        let mut vec = Vec::with_capacity(set.len());
        for line in labels.lines() {
            let allow = if set.contains(line) {
                is_allow_list
            } else {
                !is_allow_list
            };

            if allow {
                vec.push(Label::Allowed((String::from(line), None)));
            } else {
                vec.push(Label::Deny);
            }
        }

        if let Some(labels_locale) = labels_locale {
            let mut index = 0;
            let labels_locale = String::from_utf8_lossy(labels_locale);
            for line in labels_locale.lines() {
                if let Some(Label::Allowed((_, o))) = vec.get_mut(index) {
                    *o = Some(String::from(line));
                }
                index += 1;
            }
        }

        Self {
            labels: vec,
            score_threshold: option.score_threshold,
        }
    }

    #[inline(always)]
    pub fn new_category(&self, index: usize, score: f32) -> Option<Category> {
        if score >= self.score_threshold {
            if let Some(Label::Allowed((l, l_locale))) = self.labels.get(index) {
                return Some(Category {
                    index: index as i32,
                    score,
                    category_name: Some(l.clone()),
                    display_name: l_locale.clone(),
                });
            }
        }
        None
    }
}
