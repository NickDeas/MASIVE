#!/bin/bash

# MASIVE Models
python code/external_eval.py -cf code/configs/eval/ext/en_masive_mt5_emoevent_en_eval.json

python code/external_eval.py -cf code/configs/eval/ext/en_masive_mt5_goemo_eval.json

python code/external_eval.py -cf code/configs/eval/ext/en_masive_mt5_goemo_full_eval.json

python code/external_eval.py -cf code/configs/eval/ext/es_masive_mt5_emoevent_es_eval.json

# Base Models
python code/external_eval.py -cf code/configs/eval/ext/mt5_emoevent_en_eval.json

python code/external_eval.py -cf code/configs/eval/ext/mt5_goemo_eval.json

python code/external_eval.py -cf code/configs/eval/ext/mt5_goemo_full_eval.json

python code/external_eval.py -cf code/configs/eval/ext/mt5_emoevent_es_eval.json