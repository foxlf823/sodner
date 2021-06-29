
experiment_name="cadec_0001"
data_root="./data/cadec"
config_file="./training_config/cadec_working_example.jsonnet"
cuda_device=$1

ie_train_data_path=$data_root/train.json \
    ie_dev_data_path=$data_root/dev.json \
    ie_test_data_path=$data_root/test.json \
    cuda_device=$cuda_device \
    allennlp train $config_file \
    --cache-directory $data_root/cached \
    --serialization-dir ./models/$experiment_name \
    --include-package sodner \
    -f
