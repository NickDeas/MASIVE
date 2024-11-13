#!/bin/bash

# MASIVE mT5
python code/external_train.py -cf code/configs/ext_train/en_masive_t5_emoevent_en.json

python code/external_train.py -cf code/configs/ext_train/en_masive_t5_goemo.json

python code/external_train.py -cf code/configs/ext_train/en_masive_t5_goemo_full.json

# Base mT5
python code/external_train.py -cf code/configs/ext_train/t5_emoevent_en.json

python code/external_train.py -cf code/configs/ext_train/t5_goemo.json

python code/external_train.py -cf code/configs/ext_train/t5_goemo_full.json