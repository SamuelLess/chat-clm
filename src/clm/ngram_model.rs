use crate::clm::clm_model::Model;
use crate::clm::tokenizer::Token;
use crate::clm::training_options::TrainingOptions;
use std::collections::HashMap;

/// A model that implements an n-gram approach (specifically a bigram model)
/// to predict the next token based on the previous token
pub struct BigramModel {
    /// A HashMap where the key is a token, and the value is another HashMap
    /// containing the count of each token that follows it
    transition_counts: HashMap<Token, HashMap<Token, usize>>,
}

impl Model for BigramModel {
    /// Trains a bigram model by counting token pair occurrences
    fn train(tokens: Vec<Token>, _options: TrainingOptions) -> Self {
        let mut transition_counts: HashMap<Token, HashMap<Token, usize>> = HashMap::new();

        // Count bigram transitions
        for i in 0..tokens.len() - 1 {
            let current_token = &tokens[i];
            let next_token = &tokens[i + 1];

            // Update the transition count for the current token to the next token
            let next_token_counts = transition_counts
                .entry(current_token.clone())
                .or_insert_with(HashMap::new);
            *next_token_counts.entry(next_token.clone()).or_insert(0) += 1;
        }

        BigramModel { transition_counts }
    }

    /// Computes the likelihood of each possible next token based on bigram probabilities
    fn compute_likelihoods(
        &self,
        current_text: Vec<Token>,
        all_tokens: &[Token],
    ) -> HashMap<Token, f32> {
        let mut likelihoods = HashMap::new();

        // Get the last token in the current text to determine the context
        let last_token = current_text.last().unwrap();
        // Get the transition counts for the last token
        let next_token_counts = self.transition_counts.get(last_token);

        if let Some(next_token_counts) = next_token_counts {
            // Calculate the total count of all possible next tokens
            let total_count: usize = next_token_counts.values().sum();

            // Calculate the likelihood for each possible next token
            for token in all_tokens {
                let default_count = (total_count as f64 / all_tokens.len() as f64) as usize + 1;
                let count = next_token_counts.get(token).unwrap_or(&default_count);
                // Add smoothing
                likelihoods.insert(token.clone(), *count as f32 + 60.0);
            }
        } else {
            // If no transitions exist for the last token, fall back to uniform distribution
            let uniform_probability = 1.0 / all_tokens.len() as f32;
            for token in all_tokens {
                likelihoods.insert(token.clone(), uniform_probability);
            }
        }

        // Normalize the likelihoods to ensure they sum to 1.0
        let sum: f32 = likelihoods.values().sum();
        if sum > 0.0 {
            for value in likelihoods.values_mut() {
                *value /= sum;
            }
        }

        likelihoods
    }
}

/// A model that implements a unigram approach to predict the next token
/// based solely on token frequencies in the training set
pub struct UnigramModel {
    /// Total occurrences of each token for probability calculations
    token_counts: HashMap<Token, usize>,
    /// Total number of tokens in the training set
    total_tokens: usize,
}

impl Model for UnigramModel {
    /// Trains a unigram model by counting token occurrences
    fn train(tokens: Vec<Token>, _options: TrainingOptions) -> Self {
        let mut token_counts: HashMap<Token, usize> = HashMap::new();
        let total_tokens = tokens.len();

        // Count all tokens
        for token in &tokens {
            *token_counts.entry(token.clone()).or_insert(0) += 1;
        }

        UnigramModel {
            token_counts,
            total_tokens,
        }
    }

    /// Computes the likelihood of each possible next token based on unigram probabilities
    /// The unigram model ignores the context (current_text) and always returns the same
    /// probabilities based on the token frequencies in the training data
    fn compute_likelihoods(
        &self,
        _current_text: Vec<Token>,
        all_tokens: &[Token],
    ) -> HashMap<Token, f32> {
        let mut likelihoods = HashMap::new();

        if self.total_tokens > 0 {
            for token in all_tokens {
                let count = self.token_counts.get(token).unwrap();
                // Add smoothing
                likelihoods.insert(token.clone(), *count as f32 as f32);
            }
        } else {
            // Fall back to uniform distribution
            let uniform_probability = 1.0 / all_tokens.len() as f32;
            for token in all_tokens {
                likelihoods.insert(token.clone(), uniform_probability);
            }
        }

        // Normalize to ensure probabilities sum to 1.0
        let sum: f32 = likelihoods.values().sum();
        if sum > 0.0 {
            for value in likelihoods.values_mut() {
                *value /= sum;
            }
        }

        likelihoods
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigram_model_training() {
        // Create a sequence of tokens for training
        let tokens: Vec<Token> = vec![
            vec![1],
            vec![2],
            vec![3],
            vec![1],
            vec![2],
            vec![4],
            vec![1],
            vec![2],
        ];

        // Expected transitions:
        // 1 -> 2 (twice)
        // 2 -> 3 (once)
        // 3 -> 1 (once)
        // 2 -> 4 (once)
        // 4 -> 1 (once)

        let model = BigramModel::train(tokens.clone(), TrainingOptions::default());

        // Check that the model contains the correct transition counts
        assert_eq!(
            model
                .transition_counts
                .get(&vec![1])
                .unwrap()
                .get(&vec![2])
                .unwrap(),
            &3
        );
        assert_eq!(
            model
                .transition_counts
                .get(&vec![2])
                .unwrap()
                .get(&vec![3])
                .unwrap(),
            &1
        );
        assert_eq!(
            model
                .transition_counts
                .get(&vec![2])
                .unwrap()
                .get(&vec![4])
                .unwrap(),
            &1
        );
    }

    #[test]
    fn test_unigram_model_training() {
        // Create a sequence of tokens for training
        let tokens: Vec<Token> = vec![vec![1], vec![2], vec![3], vec![1], vec![2], vec![1]];

        // Expected counts:
        // 1: 3 times
        // 2: 2 times
        // 3: 1 time

        let model = UnigramModel::train(tokens.clone(), TrainingOptions::default());

        // Check that the model contains the correct counts
        assert_eq!(model.token_counts.get(&vec![1]).unwrap(), &3);
        assert_eq!(model.token_counts.get(&vec![2]).unwrap(), &2);
        assert_eq!(model.token_counts.get(&vec![3]).unwrap(), &1);
        assert_eq!(model.total_tokens, 6);
    }

    #[test]
    fn test_unigram_model_prediction() {
        // Create a sequence of tokens for training
        let tokens: Vec<Token> = vec![vec![1], vec![2], vec![3], vec![1], vec![2], vec![1]];

        let model = UnigramModel::train(tokens.clone(), TrainingOptions::default());

        // Get predictions - context should be ignored
        let all_possible_tokens = vec![vec![1], vec![2], vec![3]];
        let likelihoods = model.compute_likelihoods(vec![vec![100]], &all_possible_tokens);

        // 1 should be most likely, followed by 2, then 3
        assert!(likelihoods.get(&vec![1]).unwrap() > likelihoods.get(&vec![2]).unwrap());
        assert!(likelihoods.get(&vec![2]).unwrap() > likelihoods.get(&vec![3]).unwrap());

        // Test that the context is ignored by using a different context
        let likelihoods2 = model.compute_likelihoods(vec![vec![1]], &all_possible_tokens);
        assert_eq!(likelihoods, likelihoods2);
    }
}
