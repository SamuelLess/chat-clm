use std::collections::HashMap;

use rand::distr::{weighted::WeightedIndex, Distribution};

use super::tokenizer::{Token, Tokenizer};


pub fn print_distribution(
    tokenizer: &Tokenizer,
    distribution: &HashMap<Token, f32>,
    k: usize,
) {
    let mut sorted_distribution: Vec<_> = distribution.iter().collect();
    sorted_distribution.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let reverse_tokens = &tokenizer.build_reverse_map();

    for (token, likelihood) in sorted_distribution.iter().take(k) {
        let token_str = reverse_tokens.get(*token).unwrap();
        println!(" {:?}: {:.5}", token_str, likelihood);
    }
}

pub fn decode_top_k(distribution: &HashMap<Token, f32>, k: usize) -> Token {
    let mut sorted_distribution: Vec<_> = distribution.iter().collect();
    sorted_distribution.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let top_k = sorted_distribution.iter().take(k).map(|(k, _)| *k).collect::<Vec<_>>();

    // Sample from the top k tokens
    let mut rng = rand::rng();
    let sampler = WeightedIndex::new(top_k.iter().map(|token| distribution[*token])).unwrap();
    let sampled_index = sampler.sample(&mut rng);

    top_k[sampled_index].clone()
}
pub fn decode_top_k_unweighted(distribution: &HashMap<Token, f32>, k: usize) -> Token {
    let mut sorted_distribution: Vec<_> = distribution.iter().collect();
    sorted_distribution.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let top_k = sorted_distribution.iter().take(k).map(|(k, _)| *k).collect::<Vec<_>>();

    // Sample from the top k tokens
    let mut rng = rand::rng();
    let sampler = WeightedIndex::new(top_k.iter().map(|_| 1.0)).unwrap();
    let sampled_index = sampler.sample(&mut rng);

    top_k[sampled_index].clone()
}

pub fn decode_top_p(distribution: &HashMap<Token, f32>, p: f32) -> Token {
    let mut sorted_distribution: Vec<_> = distribution.iter().collect();
    sorted_distribution.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let mut cumulative_probability = 0.0;
    let mut selected_tokens = Vec::new();

    for (token, probability) in sorted_distribution {
        cumulative_probability += *probability;
        selected_tokens.push(token);
        if cumulative_probability >= p {
            break;
        }
    }

    // Sample from the selected tokens
    let mut rng = rand::rng();
    let sampler = WeightedIndex::new(selected_tokens.iter().map(|token| distribution[*token])).unwrap();
    let sampled_index = sampler.sample(&mut rng);

    selected_tokens[sampled_index].clone()
}