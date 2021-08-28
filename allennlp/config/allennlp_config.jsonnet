local data_dir = std.extVar("DATA_DIR");
local train_data_path = data_dir + "/train.txt";
local valid_data_path = data_dir + "/valid.txt";
local test_data_path = data_dir + "/valid.txt";
local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));
local num_epochs = std.parseInt(std.extVar("EPOCHS"));
local embedding_dim = 300;
local cuda_devices = [std.parseInt(x) for x in std.split(std.extVar("CUDA_DEVICES"), ",")];

{
  "dataset_reader": {
    "lazy": true,
    "type": "text_classification_json"
  },
  "train_data_path": train_data_path,
  "validation_data_path": valid_data_path,
  "test_data_path": test_data_path,
  "evaluate_on_test": true,
  "model": {
    "type": "basic_classifier_fbeta",
   "text_field_embedder": {
    "token_embedders": {
      "tokens": {
        "embedding_dim": embedding_dim
      }
    }
  },
    "seq2vec_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": embedding_dim,
      "hidden_size": embedding_dim,
      "num_layers": 2,
      "dropout": 0.2
    }
  },
  "iterator": {
      "type": "multiprocess",
      "base_iterator": {
          "type": "bucket",
          "batch_size": batch_size,
          "biggest_batch_first": true,
          "sorting_keys": [
              [
                  "tokens",
                  "num_tokens"
              ]
          ]
      },
        "num_workers": 2
    },
  "trainer": {
    "optimizer": {
      "type": "adam"
    },
    "cuda_device": cuda_devices,
    "patience": 5,
    "num_epochs": num_epochs,
    "validation_metric": "+fbeta",
    "should_log_learning_rate": true,
    "should_log_parameter_statistics": true,
    "num_serialized_models_to_keep": 2
  }
}

