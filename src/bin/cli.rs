use chatclm::clm::evaluate::print_top_k_tokens;
use chatclm::clm::inference::{decode_top_k_unweighted};
use chatclm::clm::training_options::TrainingOptions;
use std::collections::HashMap;
use std::io::Read;

use chatclm::clm::clm_model::{ClmModel, Model};
use chatclm::clm::tokenizer::{Token, Tokenizer};
use dotenv::dotenv;

use chatclm::clm::{save_run, uniform_model};
use clap::{Parser, Subcommand};

const MODEL_PATH: &str = "./models/";

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long, default_value_t = false)]
        use_default: bool,
    },
    Evaluate {
        // this is not optional
        model: String,
    },
    Inference {
        model: String,
    },
}

fn main() {
    dotenv().ok();
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Train { use_default }) => {
            // Implement tuning logic here
            train_model(use_default);
        }
        Some(Commands::Evaluate { model }) => {
            println!("Evaluating model: {}", model);
            eval_model(model);
        }
        Some(Commands::Inference { model }) => inference(model),
        None => {
            println!("No command provided, do something for real!");
        }
    }
}

fn load_train_tokens(training_options: &TrainingOptions, tokenizer: &Tokenizer) -> Vec<Token> {
    // read training file
    let train_text = read_file(&training_options.training_file);
    // tokenize the text
    let train_text_chars = (train_text.len() as f64 * training_options.dataset_percentage) as usize;
    let train_text = train_text[..train_text_chars].to_string();
    tokenizer.encode_fast(train_text)
}

fn train_model(use_default: &bool) {
    // read training options JSON from stdin after program start
    let training_options = if *use_default {
        TrainingOptions::default()
    } else {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line from stdin");
        serde_json::from_str(&input).expect("Failed to parse JSON")
    };

    println!("Training options: {:?}", training_options);

    let train_text = read_file(&training_options.training_file);

    // train a tokenizer
    println!("Training tokenizer...");
    let mut tokenizer = Tokenizer::new(training_options.token_byte_size);
    let tokenizer_training_count = std::cmp::min(train_text.len(), 50_000);
    tokenizer.train(&train_text[..tokenizer_training_count], training_options.token_count);
    println!("Tokenizing input...");

    let train_tokens = load_train_tokens(&training_options, &tokenizer);
    println!("Training on {} tokens", train_tokens.len());
    println!("Training model...");
    let model = ClmModel::train(train_tokens, training_options.clone());
    save_run(MODEL_PATH, &model, tokenizer.clone());
    println!("Evaluating model...");
    // evaluate the model
    let test_text = read_file(&training_options.test_file);
    let stats = chatclm::clm::evaluate::evaluate(&model, test_text, &tokenizer);
    println!("{:?}", serde_json::to_string(&stats).unwrap());
    // save the model
}

fn inference(model_name: &str) {
    // create Vec<String> for all filenames in the model directory
    let (model_files, chosen_model) = load_model(model_name);

    if let Some(file_name) = chosen_model {
        println!("Loading model: {}", file_name);
        let path = format!("{}{}", MODEL_PATH, file_name);
        let ( model, tokenizer) = chatclm::clm::load(&path);
        let all_tokens = tokenizer.get_tokens();

        println!("Prompt: ");
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line from stdin");

        let mut tokens = tokenizer.encode_fast_opt(input, true);
        loop {
            let likelihoods: HashMap<Token, f32> =
                model.compute_likelihoods(tokens.clone(), &all_tokens);
            print_top_k_tokens(&tokenizer, &likelihoods, 10);

            let next_token = decode_top_k_unweighted(&likelihoods, 1);
            tokens.push(next_token);
            let text = tokenizer.decode_with_delimiters(&tokens);
            println!("{}", text);
        }
    } else {
        println!("Model not found, available models: {:?}", model_files);
    }
}

fn eval_model(model_name: &str) {
    // create Vec<String> for all filenames in the model directory
    let (model_files, chosen_model) = load_model(model_name);
    if let Some(file_name) = chosen_model {
        let path = format!("{}{}", MODEL_PATH, file_name);
        let (mut model, tokenizer) = chatclm::clm::load(&path);
        model.options.regularization = 0.15;
        let test_text = read_file(&model.options.test_file);
        // evaluate the model
        let stats = chatclm::clm::evaluate::evaluate(&model, test_text.clone(), &tokenizer);
        println!("{:?}", serde_json::to_string(&stats).unwrap());


        // train a ngram model with the same options
        println!("Loading training tokens...");
        let training_tokens = load_train_tokens(&model.options, &tokenizer);
        
        println!("Evaluating uniform model...");
        let uniform_model = uniform_model::UniformModel::train(
            training_tokens.clone(),
            model.options.clone(),
        );
        let uniform_stats = chatclm::clm::evaluate::evaluate(&uniform_model, test_text.clone(), &tokenizer);
        println!("{:?}", serde_json::to_string(&uniform_stats).unwrap());
        println!("Evaluating bigram model...");
        let ngram_model = chatclm::clm::ngram_model::BigramModel::train(
            training_tokens.clone(),
            model.options.clone(),
        );
        let ngram_stats = chatclm::clm::evaluate::evaluate(&ngram_model, test_text.clone(), &tokenizer);
        println!("{:?}", serde_json::to_string(&ngram_stats).unwrap());

        println!("Training unigram model...");
        let unigram_model = chatclm::clm::ngram_model::UnigramModel::train(
            training_tokens,
            model.options.clone(),
        );
        let unigram_stats = chatclm::clm::evaluate::evaluate(&unigram_model, test_text, &tokenizer);
        println!("{:?}", serde_json::to_string(&unigram_stats).unwrap());
        
    } else {
        println!("Model not found, available models: {:?}", model_files);
    }
    // load the model
}

fn load_model(model_name: &str) -> (Vec<String>, Option<String>) {
    let mut model_files = Vec::new();
    let paths = std::fs::read_dir(MODEL_PATH).unwrap();
    let mut chosen_model = None;
    for path in paths {
        let path = path.unwrap().path();
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            model_files.push(file_name.to_string());
            if file_name.contains(model_name) {
                chosen_model = Some(file_name.to_string());
            }
        }
    }
    (model_files, chosen_model)
}

fn read_file(file_path: &str) -> String {
    let mut file = std::fs::File::open(file_path).expect(&format!("Could not open {}", file_path));
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read file");
    contents
}
