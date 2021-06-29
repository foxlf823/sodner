# A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition

The repo includes the code for the following [paper](https://arxiv.org/abs/2106.14373):

    @inproceedings{li2021sodner,
     title={A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition},
     author={Li, Fei and Lin, Zhichao and Zhang, Meishan and Ji, Donghong},
     booktitle={Proceedings of the ACL},
     year={2021}
    }

Setup
-----

1. Use "conda" or "virtualenv" to create a virtual python3 environment. Take "conda" as example, run:
  ```
  conda create -n sodner python=3.6
  ```
2. Activate the environment.
  ```
  conda activate sodner
  ```
2. Run the following command to install necessary packages.
  ```
  pip install -r requirements.txt
  ```
3. Download the PyTorch AllenNLP version of SciBERT from [here](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar).
Put it into the current directory.

4. Put the preprocessed data into "data" directory.
There is a sample directory for your reference to preprocess original datasets.

Training & Evaluation
-----

1. Below is the command to run experiments on the sample dataset. If use GPU, change -1 to 0 or other number that is larger than 0.
  ```
  nohup ./train_sample.sh -1 > sample_0001.log 2>&1 &
  ```

Inference
-----

1. Run the following command.
  ```
  cuda_device=-1 allennlp predict models/sample_0001/model.tar.gz data/sample/sample.json --include-package sodner --predictor my_predictor --output-file prediction.txt
  ```

Debug
-----

1. Change the settings in "sample_working_example.jsonnet" as below.
* debug: true,
* shuffle: false,

2. Add the following environment into your IDE such as PyCharm.
* ie_test_data_path=./data/sample/sample.json;
* ie_dev_data_path=./data/sample/sample.json;
* ie_train_data_path=./data/sample/sample.json;
* cuda_device=-1;

3. Run "debug_sample.py" with debug mode.

Data Preprocessing
-----

1. We show an example to preprocess the CADEC data.
First, download the code of [Dai et al. 2020](https://github.com/daixiangau/acl2020-transition-discontinuous-ner).
Use their instructions to preprocess the CADEC data and get 3 output files, namely "train.txt", "dev.txt" and "test.txt".

2. Download Stanford CoreNLP. We use "stanford-corenlp-full-2018-10-05".

3. Modify the directory paths at the beginning of "preprocess_cadec.py" based on your environment.
Create a "1.sh" file like
  ```
  #!/bin/bash
  sudo /xxx/envs/python37/bin/python "$@"
  ```
and run "1.sh preprocess_cadec.py".

Acknowledgement
-----
We thank all the people that provide their code to help us complete this project.
This project is built mainly based on the code published by [Wadden et al. 2019](https://github.com/dwadden/dygiepp).

