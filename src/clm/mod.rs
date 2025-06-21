use crate::clm::clm_model::ClmModel;
use crate::clm::tokenizer::Tokenizer;
use crate::clm::training_options::TrainingOptions;
use serde::{Deserialize, Serialize};

pub mod clm_model;
pub mod evaluate;
pub mod inference;
pub mod ngram_model;
pub mod tokenizer;
pub mod trainer;
pub mod training_options;
pub mod uniform_model;

#[derive(Clone, Serialize, Deserialize)]
pub struct SavedRun {
    pub dicts: String,
    pub tokenizer: Tokenizer,
    pub training_options: TrainingOptions,
}

pub fn save_run(base_path: &str, model: &ClmModel, tokenizer: Tokenizer) {
    // Save the model, tokenizer, and training options to the specified path
    let model_id = model
        .options
        .clone()
        .model_id
        .unwrap_or("without-id".to_string());
    println!("Saving model {} to {}", model_id, base_path);
    let saved_run = SavedRun {
        dicts: model.to_save_string(),
        tokenizer,
        training_options: model.options.clone(),
    };
    let serialized = serde_json::to_string(&saved_run).unwrap();
    // write to file
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S").to_string();
    let file_path = format!("{}/{}-{}.json", base_path, timestamp, model_id);
    std::fs::write(file_path, serialized).expect("Unable to write the file");
}

pub fn load(path: &str) -> (ClmModel, Tokenizer) {
    // Load the model, tokenizer, and training options from the specified path
    let contents = std::fs::read_to_string(path).expect("Unable to read file");
    let saved_run: SavedRun = serde_json::from_str(&contents).unwrap();
    (
        ClmModel::load_from_string(saved_run.dicts, saved_run.training_options.clone()),
        saved_run.tokenizer,
    )
}
