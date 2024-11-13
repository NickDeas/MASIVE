#!/bin/bash

# English
python code/masive_eval.py -cf code/configs/eval/en/en_masive_eval.json

python code/masive_eval.py -cf code/configs/eval/en/en_masive_t5_eval.json

# Spanish
python code/masive_eval.py -cf code/configs/eval/es/es_masive_eval.json

# Regional Spanish
python code/masive_eval.py -cf code/configs/eval/es_masive_reg_eval.json