use mediapipe_rs::tasks::text::TextClassifierBuilder;

const MODEL_1: &'static str = "assets/models/text_classification/average_word_embedding.tflite";
const MODEL_2: &'static str = "assets/models/text_classification/bert_text_classifier.tflite";

const TEXT_1: &'static str = "an imperfect but overall entertaining mystery";

#[test]
fn test_model_1() {
    text_classification_task_run(MODEL_1.into(), "1", "0");
}

#[test]
fn test_model_2() {
    text_classification_task_run(MODEL_2.into(), "positive", "negative")
}

fn text_classification_task_run(
    model_asset_path: String,
    positive_name: &str,
    negative_name: &str,
) {
    let classification_result = TextClassifierBuilder::new()
        .model_asset_path(model_asset_path)
        .finalize()
        .unwrap()
        .classify(&TEXT_1)
        .unwrap();
    assert_eq!(classification_result.classifications.len(), 1);
    let categories = &classification_result.classifications[0].categories;
    assert_eq!(categories.len(), 2);
    assert_eq!(categories[0].category_name.as_ref().unwrap(), positive_name);
    assert_eq!(categories[1].category_name.as_ref().unwrap(), negative_name);

    eprintln!("{}", classification_result);
}

#[test]
fn test_bert() {
    let classifier = TextClassifierBuilder::new()
        .model_asset_path(MODEL_2.into())
        .max_results(1)
        .finalize()
        .unwrap();
    let mut classify_session = classifier.new_session().unwrap();

    let p = "I love coding so much!";
    let n = "I don't like raining.";
    let p_result = classify_session.classify(&p).unwrap();
    let n_result = classify_session.classify(&n).unwrap();
    eprintln!("`{}` --- {}", p, p_result);
    eprintln!("`{}` --- {}", n, n_result);

    assert_eq!(p_result.classifications[0].categories[0].index, 1); // positive
    assert_eq!(n_result.classifications[0].categories[0].index, 0); // negative
}
