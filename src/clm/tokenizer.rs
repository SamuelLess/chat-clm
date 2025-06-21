use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use unidecode::unidecode;

pub type Token = Vec<u8>;

/// A BPE (Byte Pair Encoding) tokenizer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tokenizer {
    /// Tokens
    pub tokens: HashMap<String, Vec<u8>>,
    /// List of merge operations in order they were learned
    pub merges: Vec<(String, String)>,
    /// The size of the vocabulary
    pub vocab_size: usize,
    /// The size of the token in bytes
    pub token_byte_size: usize,
}

// Trie node for token prefixes
struct TrieNode {
    children: HashMap<char, TrieNode>,
    token_code: Option<Vec<u8>>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            token_code: None,
        }
    }

    // Insert a token string into the trie
    fn insert(&mut self, token: &str, code: Vec<u8>) {
        let mut node = self;
        for ch in token.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        node.token_code = Some(code);
    }
}

impl Tokenizer {
    /// Creates a new, empty tokenizer
    pub fn new(token_byte_size: usize) -> Self {
        Tokenizer {
            tokens: HashMap::new(),
            merges: Vec::new(),
            vocab_size: 0,
            token_byte_size,
        }
    }

    pub fn get_str_tokens(&self) -> Vec<String> {
        self.tokens.keys().cloned().collect()
    }

    pub fn get_tokens(&self) -> Vec<Token> {
        self.tokens.values().cloned().collect()
    }

    pub fn normalize(&self, text: &str) -> Vec<char> {
        let lowercased = unidecode(&text.to_lowercase());

        lowercased
            .to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_lowercase() || *c == ' ' || *c == '.' || *c == ',' || *c == '!')
            .collect()
    }

    /// Trains the tokenizer on the given text
    pub fn train(&mut self, text: &str, vocab_size: usize) {
        self.vocab_size = vocab_size;
        let normalized_text: Vec<char> = self.normalize(text);
        // Initialize with character-level tokens
        let mut vocab: HashMap<String, Token> = HashMap::new();
        for c in normalized_text.iter() {
            let char_str = c.to_string();
            vocab
                .entry(char_str.clone())
                .or_insert_with(|| self.compute_token_code(&char_str, self.token_byte_size));
        }

        // chunk the tokenized text in to sqrt(len) chunks
        let chunk_size = 1024;
        let mut chunks: Vec<Vec<String>> = normalized_text
            .into_iter()
            .map(|c: char| c.to_string())
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Continue merging until we reach the desired vocab size
        while vocab.len() < self.vocab_size {
            // Count pairs in the current tokenization (across all chunks)
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for chunk in &chunks {
                for i in 0..chunk.len().saturating_sub(1) {
                    if chunk[i].ends_with(' ') {
                        continue;
                    }
                    let pair = (chunk[i].clone(), chunk[i + 1].clone());
                    // Skip if first ends with a space
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            // Find the most frequent pair over all chunks
            if let Some(((first, second), _)) =
                pair_counts.into_iter().max_by_key(|&(_, count)| count)
            {
                // Create new merged token
                let new_token_str = format!("{}{}", first, second);

                let token = self.compute_token_code(&new_token_str, self.token_byte_size);

                // Add the merge to our list of merges
                self.merges.push((first.clone(), second.clone()));

                // Add the new token to our vocabulary
                vocab.insert(new_token_str.clone(), token);

                // Apply the merge to the tokenized text, chunk by chunk
                for chunk in &mut chunks {
                    let mut i = 0;
                    while i < chunk.len().saturating_sub(1) {
                        if chunk[i] == first && chunk[i + 1] == second {
                            chunk[i] = new_token_str.clone();
                            chunk.remove(i + 1);
                        } else {
                            i += 1;
                        }
                    }
                }
            } else {
                // No more merges possible
                break;
            }
        }
        // Store the final vocabulary
        self.tokens = vocab;

        /*// Create reverse mapping for decoding
        for (content, code) in &self.tokens {
            self.reverse_tokens.insert(code.clone(), content.clone());
        }*/
    }

    pub fn build_reverse_map(&self) -> HashMap<Token, String> {
        let mut reverse_tokens: HashMap<Token, String> = HashMap::new();
        for (content, code) in &self.tokens {
            reverse_tokens.insert(code.clone(), content.clone());
        }
        reverse_tokens
    }

    /// Computes a fixed-size byte code for a token based on its hash
    fn compute_token_code(&self, content: &str, token_byte_size: usize) -> Vec<u8> {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        let hash = hasher.finish();

        // Convert the hash to a fixed-size byte array
        let mut code = Vec::with_capacity(token_byte_size);
        for i in 0..token_byte_size {
            let byte = ((hash >> (i * 8)) & 0xFF) as u8;
            code.push(byte);
        }

        code
    }

    pub fn encode_fast_opt(&self, text: String, silent: bool) -> Vec<Vec<u8>> {
        // Build trie once (could be cached on self)
        if !silent {
            println!("Building trie...");
        }
        let mut root = TrieNode::new();
        for (token, code) in &self.tokens {
            root.insert(token, code.clone());
        }    
        if !silent {
            println!("Normalizing text...");
        }
        let normalized: Vec<char> = self.normalize(&text);
        drop(text);
        let mut output = Vec::new();
        let n = normalized.len();
        let mut i = 0;

        let progress_bar = indicatif::ProgressBar::new(n as u64);
        if !silent {
            progress_bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .expect("Failed to set progress bar style")
                .progress_chars("#>-"),
        );
        }
        // Traverse input greedily
        while i < n {
            let mut node = &root;
            let mut last_match_code: Option<&Vec<u8>> = None;
            let mut match_len = 0;
            // Try to extend as far as possible
            for j in i..n {
                if let Some(child) = node.children.get(&normalized[j]) {
                    node = child;
                    if let Some(code) = &node.token_code {
                        last_match_code = Some(code);
                        match_len = j - i + 1;
                    }
                } else {
                    break;
                }
            }
            // If we found a match, emit it and advance
            if let Some(code) = last_match_code {
                output.push(code.clone());
                i += match_len;
            } else {
                // No token, skip this char
                i += 1;
            }
            if !silent {
                progress_bar.set_position(i as u64);
            }
        }
        if !silent {
            progress_bar.finish_with_message("Encoding complete");
        }

        output
    }

    pub fn encode_fast(&self, text: String) -> Vec<Vec<u8>> {
        self.encode_fast_opt(text, false)
    }

    /// Decodes a sequence of token codes back into text
    pub fn decode(&self, tokens: &[Vec<u8>]) -> String {
        let mut text = String::new();

        let reverse_tokens = self.build_reverse_map();

        for token_code in tokens {
            if let Some(content) = reverse_tokens.get(token_code) {
                text.push_str(content);
            } else {
                // Handle unknown token with a placeholder
                text.push_str("[UNK]");
            }
        }

        text
    }

    pub fn decode_with_delimiters(&self, tokens: &[Vec<u8>]) -> String {
        // decodes but adds the middle dot between two tokens
        let mut text = String::new();
        let reverse_tokens = self.build_reverse_map();
        let mut first = true;
        for token_code in tokens {
            if let Some(content) = reverse_tokens.get(token_code) {
                if !first {
                    text.push('·'); // middle dot as delimiter
                }
                text.push_str(content);
                first = false;
            } else {
                // Handle unknown token with a placeholder
                if !first {
                    text.push('·');
                }
                text.push_str("[UNK]");
                first = false;
            }
        }
        text
    }

    pub fn print_token_stats(&self, tokens: &[Vec<u8>]) {
        // count token probabilities

        let mut token_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for token in tokens {
            *token_counts.entry(token.clone()).or_insert(0) += 1;
        }

        let reverse_tokens = self.build_reverse_map();

        // print it nicely formatted
        println!("Token statistics:");
        println!("Token counts: {:?}", token_counts.values().len());
        for (token, count) in token_counts.iter() {
            let token_str = reverse_tokens.get(token).unwrap();
            println!("Token: {:?}, Count: {}", token_str, count);
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(8)
    }
}
