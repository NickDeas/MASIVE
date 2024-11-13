#!/bin/bash

python code/masive_train.py -cf code/configs/train/en_train_config_mt5.json

python code/masive_train.py -cf code/configs/train/en_train_config_t5.json

python code/masive_train.py -cf code/configs/train/es_train_config.json
