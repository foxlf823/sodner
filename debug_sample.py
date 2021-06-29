
import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)


from allennlp.commands import main

experiment_name="sample_debug"
data_root="./data/sample"
config_file="./training_config/sample_working_example.jsonnet"
cuda_device=-1

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "--cache-directory", data_root+"/cached",
    "--serialization-dir", "./models/{}".format(experiment_name),
    "--include-package", 'sodner',
    '-f'
]

main()


