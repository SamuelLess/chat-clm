use crate::clm::tokenizer::Token;
use crate::clm::training_options::TrainingOptions;
use std::collections::HashMap;

use crate::clm::clm_model::Model;

/// A model that returns a uniform distribution over all tokens
pub struct UniformModel;

impl Model for UniformModel {
    /// Creates a new uniform model (ignores training data)
    fn train(_tokens: Vec<Token>, _options: TrainingOptions) -> Self {
        UniformModel
    }

    /// Returns equal probability for all tokens
    fn compute_likelihoods(
        &self,
        _current_text: Vec<Token>,
        all_tokens: &[Token],
    ) -> HashMap<Token, f32> {
        let uniform_probability = 1.0 / all_tokens.len() as f32;

        // Assign the same probability to every token
        all_tokens
            .iter()
            .map(|token| (token.clone(), uniform_probability))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        // Create some test tokens
        let tokens: Vec<Token> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        // Create the model
        let model = UniformModel::train(vec![], TrainingOptions::default());

        // Get likelihoods
        let likelihoods = model.compute_likelihoods(vec![], &tokens);

        // Check that we have the right number of tokens
        assert_eq!(likelihoods.len(), tokens.len());

        // Check that all probabilities are equal
        let expected_prob = 1.0 / tokens.len() as f32;
        for token in &tokens {
            assert_eq!(likelihoods.get(token), Some(&expected_prob));
        }

        // Ensure probabilities sum to 1.0 (accounting for floating point precision)
        let sum: f32 = likelihoods.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
