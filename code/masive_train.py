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


def parse_args():
    
    parser = argparse.ArgumentParser(
        prog = 'MASIVE Train',
        description = 'Fine-tune a T5/mT5 model on MASIVE data'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)
    
    parser.add_argument('-dr', '--data-dir',
                        type = str)
    
    parser.add_argument('-ds', '--data-suffix',
                        type = str,
                        default = '')
    
    parser.add_argument('-i', '--input-col',
                        type = str,
                        default = 'input')
    
    parser.add_argument('-t', '--target-col',
                        type = str,
                        default = 't5_trg')
    
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
    
    parser.add_argument('-ml', '--max_length',
                        type = int,
                        default = 32)
    
    parser.add_argument('--fp16',
                        type = bool,
                        default = True)
    
    parser.add_argument('-tb', '--train-batch-size',
                        type = int,
                        default = 8)
    
    parser.add_argument('-eb', '--eval-batch-size',
                        type = int,
                        default = 16)
    
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
    
    train_fp = 'train' + args['data_suffix'] + '.csv'
    val_fp   = 'val' + args['data_suffix'] + '.csv'
    
    train    = pd.read_csv(args['data_dir'] + train_fp)
    val      = pd.read_csv(args['data_dir'] + val_fp)
    
    train    = train.astype(str).drop_duplicates(subset = ['id'])
    val      = val.astype(str).drop_duplicates(subset = ['id'])
    
    train_ds = Dataset.from_pandas(train)
    val_ds   = Dataset.from_pandas(val)
        
    remove_columns = [col for col in train.columns.tolist()]
    train_tok = train_ds.map(
      tok_data(args['input_col'], args['target_col'], tokenizer),
      remove_columns=remove_columns,
      batched=True)
    
    val_tok = val_ds.map(
      tok_data(args['input_col'], args['target_col'], tokenizer),
      remove_columns=remove_columns,
      batched=True)
    
    return train_tok, val_tok

def setup_trainer(args, train_tok, val_tok, tokenizer):
    
    if 'mt5' in args['model_checkpoint']:
        model = MT5ForConditionalGeneration.from_pretrained(args['model_checkpoint'])
    else:
        model = T5ForConditionalGeneration.from_pretrained(args['model_checkpoint'])
    
    data_collator = DataCollatorForSeq2Seq(
      tokenizer,
      model=model,
      return_tensors="pt")
    
    training_args = Seq2SeqTrainingArguments(
      output_dir = args['output_dir'],
      num_train_epochs = args['num_epochs'],
      learning_rate = args['learning_rate'],
      lr_scheduler_type = "linear",
      optim = "adafactor",
      weight_decay = args['weight_decay'],
      per_device_train_batch_size = args['train_batch_size'],
      per_device_eval_batch_size = args['eval_batch_size'],
      evaluation_strategy = "epoch",
      generation_max_length = args['max_length'],
      gradient_accumulation_steps = args['grad_acc_steps'],
      gradient_checkpointing = False,
      save_total_limit = 1,
      fp16 = args['fp16'],
    )

    trainer = Seq2SeqTrainer(
      model = model,
      args = training_args,
      data_collator = data_collator,
      train_dataset = train_tok,
      eval_dataset = val_tok,
      tokenizer = tokenizer,
      preprocess_logits_for_metrics = lambda logits, _: logits,
    )

    return trainer
    
    
if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
   
    if 'mt5' in args['model_checkpoint']:
        tokenizer = MT5Tokenizer.from_pretrained(args['model_checkpoint'])
    else:
        tokenizer = T5Tokenizer.from_pretrained(args['model_checkpoint'])

    train_tok, val_tok = setup_data(args, tokenizer)
    print(f'Read Data from {args["data_dir"]}')

    trainer = setup_trainer(args, train_tok, val_tok, tokenizer)
    print(f'Setup trainer with base model {args["model_checkpoint"]}.\nBeginning training...')
    
    trainer.train()
    trainer.save_model(args['output_dir'])
    print(f'Trained and saved model to {args["output_dir"]}')
