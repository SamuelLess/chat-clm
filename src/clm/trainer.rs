use crate::clm::tokenizer::Token;
use crate::clm::training_options::TrainingOptions;
use itertools::Itertools;
use std::ffi::{c_uint, c_void};
use zstd_sys::{ZDICT_isError, ZDICT_optimizeTrainFromBuffer_fastCover};

pub fn train_model(input_tokens: &[Token], training_options: &TrainingOptions) -> Vec<u8> {
    if input_tokens.is_empty() {
        panic!("Input tokens are empty");
    }

    let chunks = input_tokens
        .chunks(training_options.training_chunk_size)
        .map(|chunk| chunk.iter().flatten().copied().collect_vec())
        .collect_vec();

    let sizes = chunks.iter().map(|x| x.len()).collect_vec();

    let raw_data = chunks.iter().flat_map(|x| x.iter()).copied().collect_vec();

    let buffer_size = std::cmp::max(
        (raw_data.len() as f64 * training_options.dictionary_size_percentage) as usize,
        256,
    );

    assert!(buffer_size >= 256, "Buffer size is too small");
    assert!(sizes.len() >= 5, "Not enough chunks to train the model");

    assert_eq!(
        sizes.iter().sum::<usize>(),
        raw_data.len(),
        "Sizes sum doesn't match raw data size"
    );

    let mut buffer = vec![0u8; buffer_size];
    let mut parameters = training_options.to_zdict_params();
    let size;
    unsafe {
        size = ZDICT_optimizeTrainFromBuffer_fastCover(
            buffer.as_mut_ptr() as *mut c_void,
            buffer_size,
            raw_data.as_ptr() as *mut c_void,
            sizes.as_ptr(),
            sizes.len() as c_uint,
            &mut parameters,
        );

        if ZDICT_isError(size) != 0 {
            panic!("Failed to train dictionary");
        }
    }
    buffer.resize(size, 0);
    buffer
}
