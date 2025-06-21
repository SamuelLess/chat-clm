use crate::clm::tokenizer::Token;
use crate::clm::trainer::train_model;
use crate::clm::training_options::TrainingOptions;
use rayon::prelude::*;
use core::panic;
use std::cmp::min;
use std::collections::HashMap;
use human_bytes::human_bytes;

pub trait Model {
    /// Trains a new model on the provided data with the given parameters
    fn train(tokens: Vec<Token>, options: TrainingOptions) -> Self;

    /// Computes the likelihood of each possible next token
    fn compute_likelihoods(
        &self,
        current_text: Vec<Token>,
        all_tokens: &[Token],
    ) -> HashMap<Token, f32>;
}

pub struct ClmModel {
    _dictionaries: Vec<Vec<u8>>,
    zstd_cdicts: Vec<*mut zstd_sys::ZSTD_CDict>,
    pub options: TrainingOptions,
}

impl Drop for ClmModel {
    fn drop(&mut self) {
        // Free all ZSTD_CDict objects when the model is dropped
        for cdict in &self.zstd_cdicts {
            unsafe {
                zstd_sys::ZSTD_freeCDict(*cdict);
            }
        }
    }
}

impl Model for ClmModel {
    fn train(tokens: Vec<Token>, options: TrainingOptions) -> Self {
        // split up the tokens into options.ensemble_size chunks
        let chunk_size = (tokens.len() as f64 / options.ensemble_size as f64)
            .ceil() as usize;

        let chunks = tokens.chunks(chunk_size);

        // Create a progress bar for training chunks
        let progress_bar = indicatif::ProgressBar::new(options.ensemble_size as u64);
        progress_bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} chunks ({msg})")
                .unwrap()
                .progress_chars("#>-")
        );

        // Train each chunk
        let chunk_results: Vec<Vec<u8>> = chunks
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(i, chunk)| {
            progress_bar.set_message(format!("Chunk {}: {}", i, human_bytes(chunk.len() as f64)));
            let dict = train_model(chunk, &options);
            progress_bar.inc(1);
            dict
            })
            .collect();

        progress_bar.finish_with_message("Training complete");

        println!("Training complete. Creating compression dictionaries...");

        let zstd_cdicts = chunk_results
            .iter()
            .map(|dict| unsafe {
                zstd_sys::ZSTD_createCDict(
                    dict.as_ptr() as *const _,
                    dict.len(),
                    options.train_compression_level as i32,
                )
            })
            .collect::<Vec<*mut zstd_sys::ZSTD_CDict>>();

        ClmModel {
            _dictionaries: chunk_results,
            zstd_cdicts,
            options,
        }
    }

    fn compute_likelihoods(
        &self,
        current_text: Vec<Token>,
        all_tokens: &[Token],
    ) -> HashMap<Token, f32> {
        let context_size = min(current_text.len(), self.options.context_window);
        let context_start = current_text.len() - context_size;
        let context = current_text[context_start..].to_vec();

        let mut scores: HashMap<Token, f64> = HashMap::new();
        
        // Initialize scores for all tokens
        for token in all_tokens.iter() {
            scores.insert(token.clone(), 0.0);
        }

        for cdict in &self.zstd_cdicts {
            let base_text = context.iter().flatten().copied().collect::<Vec<u8>>();
            let base_size = ClmModel::compress(cdict, base_text);
            for token in all_tokens.iter() {
                let mut new_text = context.clone();
                new_text.push(token.clone());
                let raw_new_text = new_text.iter().flatten().copied().collect::<Vec<u8>>();

                let compressed_size = ClmModel::compress(cdict, raw_new_text);
                
                // Add the compressed size to the token's total score
                *scores.get_mut(token).unwrap() += (compressed_size as f64 - base_size as f64) / self.zstd_cdicts.len() as f64;
            }
        }

        let inverted_scores: HashMap<Token, f64> = scores
            .iter()
            .map(|(k, v)| (k.clone(), self.options.inference_basis.powf(- v)))
            .collect();
        let sum: f64 = inverted_scores.values().cloned().sum();
        let softmax_scores: HashMap<Token, f32> = inverted_scores
            .iter()
            .map(|(k, v)| (k.clone(), (*v / sum) as f32))
            .collect();

        // regularize the scores
        let mut regularized_scores: HashMap<Token, f32> = HashMap::new();
        for (k, v) in softmax_scores.iter() {
            let regularized_score =
                v + (self.options.regularization as f32 / all_tokens.len() as f32);
            regularized_scores.insert(k.clone(), regularized_score);
        }

        // normalize the scores
        let sum: f32 = regularized_scores.values().cloned().sum();
        let softmax_scores: HashMap<Token, f32> = regularized_scores
            .iter()
            .map(|(k, v)| (k.clone(), (*v / sum)))
            .collect();

        softmax_scores
    }

    
}

impl ClmModel {
    fn compress(cdict: &*mut zstd_sys::ZSTD_CDict_s, raw_new_text: Vec<u8>) -> usize {
        let compressed_size = unsafe {
            let cctx = zstd_sys::ZSTD_createCCtx();
            if cctx.is_null() {
                panic!("Failed to create ZSTD compression context");
            }
        
            let mut dst = vec![0u8; zstd_sys::ZSTD_compressBound(raw_new_text.len())];
            let compressed_size_val = zstd_sys::ZSTD_compress_usingCDict(
                cctx,
                dst.as_mut_ptr() as *mut _,
                dst.len(),
                raw_new_text.as_ptr() as *const _,
                raw_new_text.len(),
                *cdict,
            );
        
            // Free the context before checking for errors
            zstd_sys::ZSTD_freeCCtx(cctx);
        
            // Check for errors
            if zstd_sys::ZSTD_isError(compressed_size_val) != 0 {
                panic!("Compression failed");
            } else {
                compressed_size_val
            }
        };
        compressed_size
    }
}

impl ClmModel {
    pub fn to_save_string(&self) -> String {
        serde_json::to_string(&self._dictionaries).unwrap()
    }

    pub fn load_from_string(dict_string: String, options: TrainingOptions) -> Self {
        let dictionaries: Vec<Vec<u8>> =
            serde_json::from_str(&dict_string).expect("Failed to parse dictionary string");

        let zstd_cdicts = dictionaries
            .iter()
            .map(|dict| unsafe {
                zstd_sys::ZSTD_createCDict(
                    dict.as_ptr() as *const _,
                    dict.len(),
                    options.train_compression_level as i32,
                )
            })
            .collect::<Vec<*mut zstd_sys::ZSTD_CDict>>();

        print!("Selected {} dictionaries...\r", zstd_cdicts.len());

        ClmModel {
            _dictionaries: dictionaries,
            zstd_cdicts,
            options,
        }
    }
}
