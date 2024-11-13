#!/bin/bash

# MASIVE mT5
python code/external_train.py -cf code/configs/ext_train/en_masive_mt5_emoevent_en.json

python code/external_train.py -cf code/configs/ext_train/es_masive_mt5_emoevent_es.json

python code/external_train.py -cf code/configs/ext_train/en_masive_mt5_goemo.json

python code/external_train.py -cf code/configs/ext_train/en_masive_mt5_goemo_full.json

# Base mT5
python code/external_train.py -cf code/configs/ext_train/mt5_emoevent_en.json

python code/external_train.py -cf code/configs/ext_train/mt5_emoevent_es.json

python code/external_train.py -cf code/configs/ext_train/mt5_goemo.json

python code/external_train.py -cf code/configs/ext_train/mt5_goemo_full.json