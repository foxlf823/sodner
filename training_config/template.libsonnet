// Library that accepts a parameter dict and returns a full config.

function(p) {
  local getattr(obj, attrname, default) = if attrname in obj then p[attrname] else default,

  // Storing constants.

  local validation_metrics = {
    "ner": "+ner_f1",
    "rel": "+real_ner_f1",
  },

  local display_metrics = {
    "ner": ["ner_precision", "ner_recall", "ner_f1"],
    "rel": ["ner_precision", "ner_recall", "ner_f1", "rel_precision", "rel_recall", "rel_f1", "real_ner_precision", "real_ner_recall", "real_ner_f1"],
  },

  local glove_dim = if p.debug then 50 else p.glove_dim,
  local elmo_dim = if p.debug then 256 else 1024,
  local bert_base_dim = 768,
  local bert_large_dim = 1024,

  local module_initializer = [
    [".*weight", {"type": "xavier_normal"}],
    [".*weight_matrix", {"type": "xavier_normal"}]],

  local model_initializer = [
    ["_span_width_embedding.weight", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
  ],


  ////////////////////////////////////////////////////////////////////////////////

  // Helper function.
  // Calculating dimensions.
  local use_bert = (if p.use_bert_base then true else if p.use_bert_large then true else false),

  local token_embedding_dim = ((if p.use_glove then glove_dim else 0) +
    (if p.use_char then p.char_n_filters else 0) +
    (if p.use_elmo then elmo_dim else 0) +
    (if p.use_bert_base then bert_base_dim else 0) +
    (if p.use_bert_large then bert_large_dim else 0)),

  local context_layer_output_size = (if p.use_lstm == false
    then token_embedding_dim
    else 2 * p.lstm_hidden_size),

  local endpoint_span_emb_dim = if p.use_dep then
                                ( 2 * (context_layer_output_size+p.dep_feature_dim) + p.feature_size
                                )
                                else 2 * context_layer_output_size + p.feature_size,
  local pooling_span_emb_dim = if p.use_dep then
                                ( (context_layer_output_size+p.dep_feature_dim) + p.feature_size
                                )
                                else context_layer_output_size + p.feature_size,

  local span_emb_dim =
    if p.span_extractor == "endpoint" then endpoint_span_emb_dim
    else if p.span_extractor == "pooling" then pooling_span_emb_dim
    else if p.span_extractor == "attention" then pooling_span_emb_dim
    else error "invalid span_extractor: " + p.span_extractor,

  local pair_emb_dim = 3 * span_emb_dim,
  local relation_scorer_dim = pair_emb_dim,

  ////////////////////////////////////////////////////////////////////////////////

  // Function definitions

  local make_feedforward(input_dim) = {
    input_dim: input_dim,
    num_layers: p.feedforward_layers,
    hidden_dims: p.feedforward_dim,
    activations: "relu",
    dropout: p.feedforward_dropout
  },

  // Model components

  local token_indexers = {
    [if p.use_glove then "tokens"]: {
      type: "single_id",
      lowercase_tokens: p.lowercase_token
    },
    [if p.use_char then "token_characters"]: {
      type: "characters",
      min_padding_length: 5
    },
    [if p.use_elmo then "elmo"]: {
      type: "elmo_characters"
    },
    [if use_bert then "bert"]: {
      type: "bert-pretrained",
      pretrained_model: p.bert_name+"/vocab.txt",
      do_lowercase: p.lowercase_token,
      use_starting_offsets: true
    }
  },

  local text_field_embedder = {
    [if use_bert then "allow_unmatched_keys"]: true,
    [if use_bert then "embedder_to_indexer_map"]: {
      bert: ["bert", "bert-offsets"],
      // used when glove or character enabled. since bert must be used, this works well but trick
      tokens: ["tokens"],
      token_characters: ["token_characters"],
      elmo: ["elmo"]
    },
    token_embedders: {
      [if p.use_glove then "tokens"]: {
        type: "embedding",
        pretrained_file: if p.debug then "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz" else p.glove_path,
        embedding_dim: if p.debug then 50 else p.glove_dim,
        trainable: p.tune_glove
      },
      [if p.use_char then "token_characters"]: {
        type: "character_encoding",
        embedding: {
          num_embeddings: 262,
          embedding_dim: 16
        },
        encoder: {
          type: "cnn",
          embedding_dim: 16,
          num_filters: p.char_n_filters,
          ngram_filter_sizes: [5]
        }
      },
      [if p.use_elmo then "elmo"]: {
        type: "elmo_token_embedder",
        options_file: if p.debug then "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json" else p.elmo_option,
        weight_file: if p.debug then "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5" else p.elmo_weight,
        do_layer_norm: false,
        // dont't do dropout, since we have a lexical_dropout
        dropout: 0.0,
        requires_grad: p.finetune_elmo
      },
      [if use_bert then "bert"]: {
        type: "bert-pretrained",
        pretrained_model: p.bert_name+"/weights.tar.gz",
        requires_grad: p.finetune_bert,
        scalar_mix_parameters: p.bert_scalar_mix
      }
    }
  },

  ////////////////////////////////////////////////////////////////////////////////

  // Modules

  local context_layer = (if p.use_lstm == false
    then {
      type: "pass_through",
      input_dim: token_embedding_dim
    }
    else {
      type: "stacked_bidirectional_lstm",
      input_size: token_embedding_dim,
      hidden_size: p.lstm_hidden_size,
      num_layers: p.lstm_n_layers,
      recurrent_dropout_probability: p.lstm_dropout,
      layer_dropout_probability: p.lstm_dropout
    }
  ),

  local span_extractor = {
    type: p.span_extractor,
    input_dim: span_emb_dim,
    combination: p.combination,
    num_width_embeddings: p.max_span_width,
    span_width_embedding_dim: p.feature_size
  },

  local optimizer = (if p.use_bert_base == true
    then {
        type: "bert_adam",
        lr: p.learning_rate,
        warmup: 0.1,
        t_total: 200000,
        weight_decay: p.weight_decay,
        parameter_groups: [
          [["_text_field_embedder"], {"type": "bert_adam", "lr": p.bert_learning_rate, "warmup": 0.2, "t_total": 200000, "weight_decay": p.bert_weight_decay}],
        ],
    } else {
        type: "adam",
        lr: p.learning_rate,
        weight_decay: p.weight_decay,
    }
  ),

  local model = if p.model == "sodner" then {
    type: "sodner",
    text_field_embedder: text_field_embedder,
    initializer: model_initializer,
    loss_weights: p.loss_weights,
    lexical_dropout: p.lexical_dropout,
    lstm_dropout: p.lstm_dropout,
    feature_size: p.feature_size,
    max_span_width: p.max_span_width,
    display_metrics: display_metrics[p.target],
    context_layer: context_layer,
    use_dep: p.use_dep,
    modules: {
      ner: {
        mention_feedforward: make_feedforward(span_emb_dim),
        initializer: module_initializer
      },
      relation: {
        spans_per_word: p.relation_spans_per_word,
        positive_label_weight: p.relation_positive_label_weight,
        mention_feedforward: make_feedforward(span_emb_dim),
        relation_feedforward: make_feedforward(relation_scorer_dim),
        rel_prop_dropout_A: p.rel_prop_dropout_A,
        rel_prop_dropout_f: p.rel_prop_dropout_f,
        rel_prop: p.rel_prop,
        span_emb_dim: span_emb_dim,
        initializer: module_initializer,
        use_biaffine_rel: p.use_biaffine_rel,
      },
      gat_tree: {
        span_emb_dim: context_layer_output_size,
        tree_prop: p.gcn_layer,
        initializer: module_initializer,
        tree_dropout: p.gcn_dropout,
        feature_dim: p.dep_feature_dim,
        aggcn_heads: p.aggcn_heads,
        aggcn_sublayer_first: p.aggcn_sublayer_first,
        aggcn_sublayer_second: p.aggcn_sublayer_second,
      },
    },
    span_extractor: span_extractor
  }
  else error "invalid model: " + p.model,

  ////////////////////////////////////////////////////////////////////////////////


  // The model

  random_seed: getattr(p, "random_seed", 13370),
  numpy_seed: getattr(p, "numpy_seed", 1337),
  pytorch_seed: getattr(p, "pytorch_seed", 133),
  dataset_reader: {
    type: "ie_json",
    token_indexers: token_indexers,
    max_span_width: p.max_span_width,
    context_width: p.context_width,
    debug: getattr(p, "debug", false),
    use_overlap_rel: p.use_overlap_rel,
  },
  train_data_path: std.extVar("ie_train_data_path"),
  validation_data_path: std.extVar("ie_dev_data_path"),
  test_data_path: std.extVar("ie_test_data_path"),
  model: model,
  iterator: {
    type: "ie_batch",
    batch_size: p.batch_size,
    [if "instances_per_epoch" in p then "instances_per_epoch"]: p.instances_per_epoch
  },
  validation_iterator: {
    type: "ie_document",
    batch_size: p.batch_size
  },
  trainer: {
    num_serialized_models_to_keep: 1,
    num_epochs: p.num_epochs,
    grad_norm: 5.0,
    patience : p.patience,
    cuda_device : [std.parseInt(x) for x in std.split(std.extVar("cuda_device"), ",")],
    validation_metric: validation_metrics[p.target],
    learning_rate_scheduler: p.learning_rate_scheduler,
    optimizer: optimizer,
    [if "moving_average_decay" in p then "moving_average"]: {
      type: "exponential",
      decay: p.moving_average_decay
    },
    shuffle: p.shuffle
  },
  evaluate_on_test: getattr(p, "evaluate_on_test", false),
}
