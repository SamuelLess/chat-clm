use std::time::Duration;

use crate::clm::clm_model::Model;
use crate::clm::tokenizer::Tokenizer;
use indicatif::{ProgressBar, ProgressStyle};
use num::Signed;
use serde::{Deserialize, Serialize};

use crate::clm::tokenizer::Token;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub average_likelihood: f64,
    pub cross_entropy: f64,
    pub perplexity: f64,
    pub perplexity_stderr: f64,
    pub time_per_token: f64,
    pub ppt: f64,
    pub ppt_stderr: f64,
}

pub fn check_distribution<T>(likelihoods: &std::collections::HashMap<T, f32>) {
    let mut total = 0.0;
    for (_, likelihood) in likelihoods.iter() {
        if likelihood.is_infinite() || likelihood.is_nan() || likelihood.is_negative() {
            panic!("Likelihood is infinite, NaN, or negative");
        }

        total += *likelihood as f64;
    }
    if (total - 1.0).abs() > 0.001 {
        println!("Warning: Likelihoods do not sum to 1.0");
    }
}

pub fn print_top_k_tokens(
    tokenizer: &Tokenizer,
    likelihoods: &std::collections::HashMap<Token, f32>,
    k: usize,
) {
    let mut sorted_likelihoods: Vec<_> = likelihoods.iter().collect();
    sorted_likelihoods.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let reverse_tokens = &tokenizer.build_reverse_map();

    println!("Top {} tokens:", k);
    for (token, likelihood) in sorted_likelihoods.iter().take(k) {
        let token = *token;
        let token_str = reverse_tokens.get(token).unwrap();
        println!("Token: {:?}, Likelihood: {}", token_str, likelihood);
    }
}
/// Evaluates a model implementing the Model trait on the given text
pub fn evaluate<M: Model>(model: &M, text: String, tokenizer: &Tokenizer) -> ModelStats {
    let tokens = tokenizer.encode_fast(text);

    let progress_bar = ProgressBar::new((tokens.len() as u64).saturating_sub(1));
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {eta} rem. ({msg})",
            )
            .expect("Failed to set progress bar style")
            .progress_chars("#>-"),
    );
    progress_bar.inc(1);

    let all_tokens = tokenizer.get_tokens();
    let time = std::time::Instant::now();

    let positions: Vec<usize> = (32..tokens.len()).collect();
    let mut likelihoods: Vec<f64> = Vec::with_capacity(positions.len());
    for &pos in positions.iter() {
        let current_text = tokens[0..(pos - 1)].to_vec();
        let ground_truth = tokens[pos].clone();

        let token_likelihoods = model.compute_likelihoods(current_text, &all_tokens);
        check_distribution(&token_likelihoods);

        let ground_truth_likelihood = token_likelihoods.get(&ground_truth)
            .unwrap_or_else(|| panic!("Ground truth token not found in likelihoods!"));
        progress_bar.inc(1);
        let stats = calculate_model_stats(&likelihoods, Duration::from_micros(1), &all_tokens);
        progress_bar.set_message(format!("ppt: {:.2}", stats.ppt));
        likelihoods.push(*ground_truth_likelihood as f64);
    }

    let elapsed_time = time.elapsed();
    progress_bar.finish_and_clear();

    let model_stats = calculate_model_stats(&likelihoods, elapsed_time, &all_tokens);
    model_stats
}

/// Calculates statistics for model evaluation from the token likelihoods
fn calculate_model_stats(likelihoods: &[f64], elapsed_time: std::time::Duration, all_tokens: &[Token]) -> ModelStats {
    let average_likelihood = likelihoods.iter().sum::<f64>() / likelihoods.len() as f64;

    let cross_entropies = likelihoods.iter().map(|&x| -x.ln()).collect::<Vec<_>>();
    let cross_entropy_mean = cross_entropies.iter().sum::<f64>() / (cross_entropies.len()) as f64;
    
    let cross_entropy_variance = cross_entropies.iter().map(|&x| (x - cross_entropy_mean).powi(2)).sum::<f64>() / (cross_entropies.len()-1) as f64;
    let cross_entropy_stderr = cross_entropy_variance.sqrt() / (cross_entropies.len() as f64).sqrt();

    let perplexity = cross_entropy_mean.exp();
    let perplexity_stderr = perplexity * cross_entropy_stderr;

    let ppt = perplexity / all_tokens.len() as f64;
    let ppt_stderr = perplexity_stderr / all_tokens.len() as f64;

    ModelStats {
        average_likelihood,
        cross_entropy: cross_entropy_mean,
        perplexity,
        perplexity_stderr,
        time_per_token: elapsed_time.as_secs_f64() / likelihoods.len() as f64,
        ppt,
        ppt_stderr,
    }
}
