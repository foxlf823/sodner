
experiment_name="sample_0001"
data_root="./data/sample"
config_file="./training_config/sample_working_example.jsonnet"
cuda_device=$1

ie_train_data_path=$data_root/sample.json \
    ie_dev_data_path=$data_root/sample.json \
    ie_test_data_path=$data_root/sample.json \
    cuda_device=$cuda_device \
    allennlp train $config_file \
    --cache-directory $data_root/cached \
    --serialization-dir ./models/$experiment_name \
    --include-package sodner \
    -f
