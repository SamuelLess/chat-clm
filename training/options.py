DEFAULT_TRAINING_OPTIONS = {
    "d": 6,
    "f": 9,
    "k": 12,
    "steps": 0,
    "nb_threads": 8,
    "split_point": 1.0,
    "accel": 1,
    "shrink_dict": 0,
    "shrink_dict_max_regression": 3,
    "train_compression_level": 1,
    "dictionary_size_percentage": 0.05,
    "ensemble_size": 1,
    "training_chunk_size": 64,
    "token_count": 120,
    "token_byte_size": 6,
    "context_window": 20,
    "dataset_percentage": 0.9,
    "dictionaries_taken": 0.25,
    "regularization": 10.0,
    "training_file": "enwik8.txt",
    "test_file": "test.txt",
}

SWEEP_CONFIG = {
    "method": "grid",
    "name": "ensemble-sweep",
    "metric": {
        "goal": "minimize",
        "name": "ppt",
    },
    "parameters": {
        "d": {
            "value": 8,
        },
        "f": {
            "value": 16,        
        },
        "k": {
            "value": 6078,        
        },
        "steps": {
            "value": 0,
        },
        "nb_theads": {
            "value": 1,
        },
        "split_point": {
            "value": 1.0,
        },
        "accel": {
            "value": 1,
        },
        "shrink_dict": {
            "value": 1,
        },
        "shrink_dict_max_regression": {
            "value": 3,
        },
        "train_compression_level": {
            "value": 21,
        },
        "dictionary_size_percentage": {
            "value": 0.07,
        },
        "ensemble_size": {
            "values": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150, 200],
        },
        "training_chunk_size": {
            "value": 256,
        },
        "token_count": {
            "value": 220,
        },
        "token_byte_size": {
            "value": 5,
        },
        "context_window": {
            "value": 32,
        },
        "dataset_percentage": {
            "value": 0.95,
        },
        "regularization": {
            "value": 0,
        },
        "inference_basis": {
            "value": 1.55,
        },
    },
}
