use std::ffi::c_int;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingOptions {
    pub d: u32,
    pub f: u32,
    pub k: u32,
    pub steps: u32,
    pub nb_threads: u32,
    pub split_point: f64,
    pub accel: u32,
    pub shrink_dict: u32,
    pub shrink_dict_max_regression: u32,
    pub train_compression_level: i32,
    pub dictionary_size_percentage: f64, // 0.0 to 1.0, how big the dictionary should be compared to the input data
    pub ensemble_size: usize,            // number of models to train
    pub training_chunk_size: usize, // how many tokens to put in a chunk for training the dictionary
    pub token_count: usize,         // how many tokens to use
    pub token_byte_size: usize,     // how many bytes to use for each token
    pub context_window: usize,      // how many tokens to look back during prediction
    pub dataset_percentage: f64,    // how much of the dataset to use for training
    pub regularization: f64,     // how much to regularize the model
    pub model_id: Option<String>, // model id for the model
    pub training_file: String,   // file to use for training
    pub test_file: String,       // file to use for testing
    pub inference_basis: f64,    // basis in probability space for inference
}

impl TrainingOptions {
    pub fn to_zdict_params(&self) -> zstd_sys::ZDICT_fastCover_params_t {
        zstd_sys::ZDICT_fastCover_params_t {
            k: self.k,
            d: self.d,
            f: self.f,
            steps: self.steps,
            nbThreads: self.nb_threads,
            splitPoint: self.split_point,
            accel: self.accel,
            shrinkDict: self.shrink_dict,
            shrinkDictMaxRegression: self.shrink_dict_max_regression,
            zParams: zstd_sys::ZDICT_params_t {
                compressionLevel: self.train_compression_level as c_int,
                notificationLevel: 1,
                dictID: 0,
            },
        }
    }
}

impl Default for TrainingOptions {
    fn default() -> Self {
        TrainingOptions {
            d: 8,
            f: 16,
            k: 6078,
            steps: 0,
            nb_threads: 12,
            split_point: 1.0,
            accel: 1,
            shrink_dict: 1,
            shrink_dict_max_regression: 3,
            train_compression_level: 21,
            dictionary_size_percentage: 0.08,
            ensemble_size: 15,
            training_chunk_size: 256,
            token_count: 210,
            token_byte_size: 5,
            context_window: 32,
            dataset_percentage: 1.0,
            regularization: 0.0,
            model_id: Some(String::from("enwik9_token_size_6")),
            training_file: String::from("data/enwik9"),
            test_file: String::from("test.txt"),
            inference_basis: 1.55,
        }
    }
}
