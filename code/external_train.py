import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import re
import json
from ast import literal_eval
from collections import defaultdict
import random

import transformers
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import datasets
from datasets import Dataset

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score,  recall_score, precision_score, multilabel_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import argparse

# Set Seeds
SEED = 13
random.seed(SEED) # Python
np.random.seed(SEED) # Numpy
torch.manual_seed(SEED) # Torch
transformers.set_seed(SEED) # Transformers

# Label Constants
LANG_DICT   = {'goemo': 'en',
               'goemo_full': 'en',
               'emoevent_en': 'en',
               'emoevent_es': 'es'}

LABELS_DICT = {'goemo': ['angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'nothing'],
               'goemo_full': ['admiration', 'amused', 'angry', 'annoyed', 'approving', 'caring', 'confused', 'curious', 'desire', 'disappointed', 'disapproval', 'disgusted', 'embarassed', 'excited', 'afraid', 'grateful', 'grief', 'happy', 'love', 'nervous', 'optimistic', 'proud', 'realized', 'relieved', 'remorseful', 'sad', 'surprised', 'nothing'],
               'emoevent_en': ['angry', 'disgusted', 'scared', 'happy', 'sad', 'surprised', 'nothing'],
               'emoevent_es': ['enojado', 'desagradado', 'asustado', 'feliz', 'triste', 'sorprendido', 'nada']
              }

# File Names
FP_DICT = {'goemo': 'goemo_ekman_train.csv',
           'goemo_full': 'goemo_full_train.csv',
           'emoevent_en': 'emo_event_en_train.csv',
           'emoevent_es': 'emo_event_es_train.csv'}

def parse_args():
    
    parser = argparse.ArgumentParser(
        prog = 'MASIVE External Train',
        description = 'Fine-tune a T5/mT5 model on prior emotion detection benchmarks.'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)
    
    parser.add_argument('-dd', '--data-dir',
                        type = str)
    
    parser.add_argument('-td', '--train-data',
                        type = str,
                        default = 'goemo')
    
    parser.add_argument('-i', '--input-col',
                        type = str,
                        default = 'text')
    
    parser.add_argument('-e', '--num-epochs',
                        type = int,
                        default = 3)
    
    parser.add_argument('-lr', '--learning-rate',
                        type = float,
                        default = 1e-4)
    
    parser.add_argument('-gs', '--grad-acc-steps',
                        type = int,
                        default = 1)
    
    parser.add_argument('-wd', '--weight-decay',
                        type = float,
                        default = 0.01)
    
    parser.add_argument('-ml', '--max-length',
                        type = int,
                        default = 64)
    
    parser.add_argument('--fp16',
                        type = bool,
                        default = True)
    
    parser.add_argument('-tb', '--train-batch-size',
                        type = int,
                        default = 48)
    
    parser.add_argument('-mc', '--model-checkpoint',
                        type = str)
    
    parser.add_argument('-o', '--output-dir',
                        type = str)
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
    
    return args


def tokenize_sample_data(data, src, trg, tokenizer):
    input_feature = tokenizer(data[src], truncation=True, max_length = 512, padding = 'longest')
    label         = tokenizer(data[trg], truncation=True, max_length = 128, padding = 'longest')

    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }

def tok_data(src, trg, tokenizer):
    return lambda data: tokenize_sample_data(data, src, trg, tokenizer)

    
def setup_data(args, tokenizer):
   
    train = pd.read_csv(args['data_dir'] + '/' + FP_DICT[args['train_data']])
    train = train.astype(str)
    
    
    ds_labels = LABELS_DICT[args['train_data']]
    lang      = LANG_DICT[args['train_data']]
    train = format_df(args, train, ds_labels, lang)
    
    train_ds = Dataset.from_pandas(train)
        
    remove_columns = [col for col in train.columns.tolist()]
    train_tok = train_ds.map(
      tok_data('input', 'ground', tokenizer),
      remove_columns=remove_columns,
      batched=True)
    
    return train_tok

def format_df(args, df, ds_labels, lang):
    data = df.copy()
    data = data[['id', 'text', 'label']]
    data['label_text'] = data['label'].apply(lambda labels: [ds_labels[int(label)] for label in str(labels).split(',')]) 
    
    #English
    if lang == 'en':
        data['input'] = data[args['input_col']] + ' I feel ' + data['label_text'].apply(lambda l: ' and '.join([f'<extra_id_{i}> {label}' for i, label in enumerate(l)]))
    #Spanish
    elif lang == 'es':
        data['input'] = data[args['input_col']] + ' Me siento ' + data['label_text'].apply(lambda l: ' y '.join([f'<extra_id_{i}> {label}' for i, label in enumerate(l)]))

    data['ground'] = ' <extra_id_0> ' + data['label_text'].apply(lambda lab_l: lab_l[0])
    
    print(data[data['ground'].apply(len) > 20].iloc[0]['ground'])
    
    return data

def setup_trainer(args, train_tok, tokenizer):
    
    if 'mt5' in args['model_checkpoint']:
        model = MT5ForConditionalGeneration.from_pretrained(args['model_checkpoint'])
    else:
        model = T5ForConditionalGeneration.from_pretrained(args['model_checkpoint'])
    
    data_collator = DataCollatorForSeq2Seq(
      tokenizer,
      model=model,
      return_tensors="pt")
    
    training_args = Seq2SeqTrainingArguments(
      output_dir = get_save_path(args),
      num_train_epochs = args['num_epochs'],
      learning_rate = args['learning_rate'],
      lr_scheduler_type = "linear",
      optim = "adafactor",
      weight_decay = args['weight_decay'],
      per_device_train_batch_size = args['train_batch_size'],
      evaluation_strategy = "no",
      generation_max_length = args['max_length'],
      gradient_accumulation_steps = args['grad_acc_steps'],
      gradient_checkpointing = False,
      save_total_limit = 1,
      do_eval = False,
      fp16 = args['fp16'],
    )

    # Normal Training
    trainer = Seq2SeqTrainer(
      model = model,
      args = training_args,
      data_collator = data_collator,
      train_dataset = train_tok,
      tokenizer = tokenizer,
      preprocess_logits_for_metrics = lambda logits, _: logits,
    )
   

    return trainer
    
def get_save_path(args):
    return f'{args["output_dir"]}/{args["model_checkpoint"].split("/")[-1].replace("-", "_") if args["model_checkpoint"][-1] != "/" else args["model_checkpoint"].split("/")[-2].replace("-", "_")}_{args["train_data"]}'
    
if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
   
    if 'mt5' in args['model_checkpoint']:
        tokenizer = MT5Tokenizer.from_pretrained(args['model_checkpoint'])
    else:
        tokenizer = T5Tokenizer.from_pretrained(args['model_checkpoint'])

    train_tok = setup_data(args, tokenizer)
    print(f'Read Data from {args["data_dir"]}')

    trainer = setup_trainer(args, train_tok, tokenizer)
    print(f'Setup trainer with base model {args["model_checkpoint"]}.')
    print('Beginning training...')
    
    trainer.train()
    trainer.save_model(get_save_path(args))
    print(f'Trained and saved model to {get_save_path(args)}')
