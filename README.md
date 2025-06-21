# chatCLM - ZSTD Compression-Based Language Model

## Overview

chatCLM (Compression Language Model) is an experimental demonstration that explores the fundamental connection between lossless compression and language modeling. Based on the principle that "compression is prediction," this work demonstrates how file compression algorithms can be used for next-token prediction in language modeling.

It is described and analysed in detail in [Compression is Prediction: Creating a Language Model Based on Lossless Compression](Paper.pdf).

### Results

The model achieves a perplexity of 183.9 on the enwik9 dataset, significantly better than a uniform model (220 perplexity) but not competitive with modern language models. While the generated text lacks coherence, it demonstrates meaningful structure learning and serves as an academically interesting illustration of the compression-prediction equivalence.

This is an **interesting demonstration rather than a practical language modeling approach**. The goal is to explore the theoretical connection between information theory and intelligence, not to create a competitive language model.


## Features

- **Compression-based prediction**: Uses ZSTD compression dictionaries for likelihood estimation
- **Ensemble modeling**: Trains multiple models on different data chunks
- **Custom tokenization**: BPE tokenizer with configurable vocabulary size
- **Multiple model types**: CLM, n-gram models (unigram, bigram), and uniform baseline
- **Comprehensive evaluation**: Built-in evaluation metrics and comparison tools
- **Interactive inference**: Real-time text generation with top-k token display

## Installation

### Prerequisites

- Rust and Cargo
### Building

```bash
git clone https://github.com/SamuelLess/chat-clm.git
cd chat-clm
cargo build --release
```

## Usage

The CLI tool provides three main commands: `train`, `evaluate`, and `inference`.

### Training a Model

Train a model with default parameters:

```bash
./target/release/cli train --use-default
```

To train a model with custom parameters, run the command without the `--use-default` flag and pass a parameter object as JSON on stdin.

### Model Evaluation

Evaluate a trained model and compare with baseline models:

```bash
./target/release/cli evaluate enwik9
```

This command will:
- Load the specified model (searches for files containing "enwik9" in the models directory)
- Evaluate the CLM model on the test dataset
- Train and evaluate baseline models (uniform, unigram, bigram) for comparison
- Output performance statistics in JSON format

To run the command, you need to unpack the trained model in the `./models/` directory and create a test file names `test.txt` with the evaluation text.

### Interactive Inference

Generate text interactively with a trained model:

```bash
./target/release/cli inference enwik9
```

This will:
- Load the specified model
- Prompt for input text
- Generate and display text token by token

To run this command you either need to train a model first or unpack the trained model in the `./models/` directory.

## Training Parameters

Key training parameters include:

- **ensemble_size**: Number of compression dictionaries to train (default: 15)
- **token_count**: Vocabulary size for the tokenizer (default: 210)
- **token_byte_size**: Byte size for token encoding (default: 5)
- **context_window**: Number of previous tokens to consider (default: 32)
- **dictionary_size_percentage**: Size of compression dictionary relative to input (default: 0.08)
- **train_compression_level**: ZSTD compression level for training (default: 21)
- **inference_basis**: Base for exponential transformation during inference (default: 1.55)

The other parameters are found in `src/clm/training_options.rs` and can be adjusted as needed through a config object passed via stdin during training.

## Model Output

Trained models are saved in the `./models/` directory with JSON format containing:
- Compression dictionaries
- Tokenizer vocabulary and merges
- Training configuration parameters

## License

This project was developed by Niels Glodny, Samuel Lessmann and Niclas Dern.

It is licensed under the GNU General Public License v3.0. 

