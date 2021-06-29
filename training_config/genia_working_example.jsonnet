// Import template file.

local template = import "template.libsonnet";

////////////////////

// Set options.

local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local params = {
  // ner or rel
  target: "ner",

  // debug mode or not
  debug: false,

  model: "sodner",

  // use glove or not
  use_glove: false,
  tune_glove: true,
  glove_path: "./glove.6B.100d.txt",
  glove_dim: 100,

  // use character feature or not
  use_char: false,
  char_n_filters: 50,

  // use elmo or not
  use_elmo: false,
  elmo_option: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
  elmo_weight: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
  finetune_elmo: true,

  // use bert or not
  use_bert_base: true,
  use_bert_large: false,
  finetune_bert: true,
  bert_name: "./scibert_scivocab_cased",
  bert_scalar_mix: null,

  // lowercase token indexer
  lowercase_token: false,

  // dropout on the word representations
  lexical_dropout: 0.5,

  // how many sentences to use
  context_width: 1,

  // lstm on top of word representation layer
  use_lstm: false,
  lstm_hidden_size: 200,
  lstm_n_layers: 1,
  lstm_dropout: 0.0,

  // dep-based GCN
  use_dep: true,
  gcn_layer: 2,
  gcn_dropout: 0.2,
  dep_feature_dim: 64,
  aggcn_heads: 4,
  aggcn_sublayer_first: 2,
  aggcn_sublayer_second: 4,

  // span representation: endpoint, pooling
  span_extractor: "endpoint",
  combination: "x,y",
  max_span_width: 8,
  feature_size: 20,

  // output layer MLP
  feedforward_layers: 2,
  feedforward_dim: 150,
  feedforward_dropout: 0.4,

  // relation module config
  rel_prop: 1,
  rel_prop_dropout_A: 0.0,
  rel_prop_dropout_f: 0.0,
  relation_spans_per_word: 0.5,
  relation_positive_label_weight: 1.0,
  use_overlap_rel: true,
  use_biaffine_rel: false,

  // training config
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
  },
  batch_size: 2,
  instances_per_epoch: 1000,
  num_epochs: 250,
  patience: 15,
  learning_rate_scheduler:  {
    type: "reduce_on_plateau",
    factor: 0.5,
    mode: "max",
    patience: 4
  },
  shuffle: true,
  evaluate_on_test: true,

  learning_rate: 1e-3,
  weight_decay: 0.0,
  bert_learning_rate: 5e-5,
  bert_weight_decay: 0.01,

};

////////////////////

// Feed options into template.

template(params)
