#!/bin/bash

# MASIVE Models
python code/external_eval.py -cf code/configs/eval/ext/en_masive_t5_emoevent_en_eval.json

python code/external_eval.py -cf code/configs/eval/ext/en_masive_t5_goemo_eval.json

python code/external_eval.py -cf code/configs/eval/ext/en_masive_t5_goemo_full_eval.json

# Base Models
python code/external_eval.py -cf code/configs/eval/ext/t5_emoevent_en_eval.json

python code/external_eval.py -cf code/configs/eval/ext/t5_goemo_eval.json

python code/external_eval.py -cf code/configs/eval/ext/t5_goemo_full_eval.json
