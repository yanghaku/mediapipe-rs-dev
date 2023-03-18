pub(crate) struct ClassifierBuilder {
    /// The maximum number of top-scored classification results to return. If < 0,
    /// all available results will be returned. If 0, an invalid argument error is
    /// returned.
    pub max_results: i32,

    /// Score threshold to override the one provided in the model metadata (if
    /// any). Results below this value are rejected.
    pub score_threshold: f32,

    /// The locale to use for display names specified through the TFLite Model
    /// Metadata, if any. Defaults to English.
    pub display_names_locale: String,

    /// The allow list of category names. If non-empty, detection results whose
    /// category name is not in this set will be filtered out. Duplicate or unknown
    /// category names are ignored. Mutually exclusive with category_deny_list.
    pub category_allow_list: Vec<String>,

    /// The deny list of category names. If non-empty, detection results whose
    /// category name is in this set will be filtered out. Duplicate or unknown
    /// category names are ignored. Mutually exclusive with category_allow_list.
    pub category_deny_list: Vec<String>,
}

impl Default for ClassifierBuilder {
    fn default() -> Self {
        Self {
            display_names_locale: "en".into(),
            max_results: -1,
            score_threshold: 0.0f32,
            category_allow_list: Vec::new(),
            category_deny_list: Vec::new(),
        }
    }
}

macro_rules! classifier_builder_impl {
    () => {
        #[inline(always)]
        pub fn display_names_locale(mut self, display_names_locale: String) -> Self {
            self.classifier_builder.display_names_locale = display_names_locale;
            self
        }

        #[inline(always)]
        pub fn max_results(mut self, max_results: i32) -> Self {
            self.classifier_builder.max_results = max_results;
            self
        }

        #[inline(always)]
        pub fn score_threshold(mut self, score_threshold: f32) -> Self {
            self.classifier_builder.score_threshold = score_threshold;
            self
        }

        #[inline(always)]
        pub fn category_allow_list(mut self, category_allow_list: Vec<String>) -> Self {
            self.classifier_builder.category_allow_list = category_allow_list;
            self
        }

        #[inline(always)]
        pub fn category_deny_list(mut self, category_deny_list: Vec<String>) -> Self {
            self.classifier_builder.category_deny_list = category_deny_list;
            self
        }
    };
}

macro_rules! classifier_builder_check {
    ( $self:ident ) => {{
        if $self.classifier_builder.max_results == 0 {
            return Err(crate::Error::ArgumentError(
                "The number of max results cannot be zero".into(),
            ));
        }
        if !$self.classifier_builder.category_allow_list.is_empty()
            && !$self.classifier_builder.category_deny_list.is_empty()
        {
            return Err(crate::Error::ArgumentError(
                "Cannot use both `category_allow_list` and `category_deny_list`".into(),
            ));
        }
    }};
}

macro_rules! classifier_build_info_get_impl {
    () => {
        #[inline(always)]
        pub fn max_result(&self) -> i32 {
            self.build_info.classifier_builder.max_results
        }

        #[inline(always)]
        pub fn score_threshold(&self) -> f32 {
            self.build_info.classifier_builder.score_threshold
        }

        #[inline(always)]
        pub fn display_names_locale(&self) -> &String {
            &self.build_info.classifier_builder.display_names_locale
        }

        #[inline(always)]
        pub fn category_allow_list(&self) -> &Vec<String> {
            &self.build_info.classifier_builder.category_allow_list
        }

        #[inline(always)]
        pub fn category_deny_list(&self) -> &Vec<String> {
            &self.build_info.classifier_builder.category_deny_list
        }
    };
}
