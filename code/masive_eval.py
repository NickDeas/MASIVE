import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertModel
from transformers import DataCollatorForSeq2Seq
from transformers import LogitsProcessorList, LogitsProcessor
import datasets
from datasets import Dataset

import os
import re
import random
import json
from ast import literal_eval
from collections import defaultdict
import argparse
import itertools
import time

from sklearn.metrics import f1_score,  recall_score, precision_score, multilabel_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from huggingface_hub import login

from metrics import *

# Set Seeds
SEED = 13
random.seed(SEED) # Python
np.random.seed(SEED) # Numpy
torch.manual_seed(SEED) # Torch
transformers.set_seed(SEED) # Transformers

def parse_args():
    
    parser = argparse.ArgumentParser(
        prog = 'MASIVE Eval',
        description = 'Evaluate a T5/mT5 model on the full MASIVE test set.'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)

    parser.add_argument('-mc', '--model-checkpoint',
                        type = str)

    parser.add_argument('-ec', '--embed-checkpoint',
                        type = str,
                        default = 'bert-base-multilingual-uncased')
    
    parser.add_argument('-t', '--test-fp',
                        type = str)
    
    parser.add_argument('-i', '--input-col',
                        type = str,
                        default = 'clip_input')
    
    parser.add_argument('-tc', '--target-col',
                        type = str,
                        default = 't5_trg')

    parser.add_argument('-st', '--sim-type',
                        type = str,
                        choices = ['min', 'max', 'avg'],
                        default = 'max')
    
    parser.add_argument('-b', '--batch-size',
                        type = int,
                        default = 16)
    
    parser.add_argument('-r', '--results-dir',
                        type = str)

    parser.add_argument('-d', '--device',
                        type = int,
                        default = 0)
    
    parser.add_argument('-ed', '--embed-device',
                        type = int,
                        default = None)
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
            
    if args['embed_device'] is None:
        args['embed_device'] = args['device']
    
    return args

def get_device(args):
    return torch.device(f'cuda:{args["device"]}')

def get_embed_device(args):
    return torch.device(f'cuda:{args["embed_device"]}')

def load_embed_model(args):
    tokenizer = BertTokenizer.from_pretrained(args['embed_checkpoint'])
    bert_model = BertModel.from_pretrained(args['embed_checkpoint']).to(get_embed_device(args))
    bert_model.eval()
    return bert_model, tokenizer

def load_model(args):
    
    if 'mt5' in args['model_checkpoint']:
        tok   = MT5Tokenizer.from_pretrained(args['model_checkpoint'])
        model = MT5ForConditionalGeneration.from_pretrained(args['model_checkpoint']).to(get_device(args))
    else:
        tok   = T5Tokenizer.from_pretrained(args['model_checkpoint'])
        model = T5ForConditionalGeneration.from_pretrained(args['model_checkpoint']).to(get_device(args))

    model.eval()
    return model, tok

def tokenize_sample_data(data, src, trg, tokenizer):

    input_feature = tokenizer(data[src], truncation=True, max_length = 511, padding = 'longest')
    label         = tokenizer(data[trg], truncation=True, max_length = 511, padding = 'longest')

    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }

def tok_data(args, src, trg, tokenizer):
    return lambda data: tokenize_sample_data(data, src, trg, tokenizer)
    
def setup_data(args, model, tokenizer): 

    test = pd.read_csv(args['test_fp'])
    test = test.astype(str).drop_duplicates(subset = 'id')
    
    test_ds = Dataset.from_pandas(test)

    REMOVE_COLS = [col for col in test.columns]
    test_tok = test_ds.map(
       tok_data(args, args['input_col'], args['target_col'], tokenizer),
       remove_columns=REMOVE_COLS,
       batched=True)

    data_collator = DataCollatorForSeq2Seq(
      tokenizer,
      model=model,
      return_tensors="pt",
      padding=True)

    test_dl = DataLoader(test_tok, 
                    batch_size = args['batch_size'], 
                    collate_fn = data_collator,
                    shuffle = False,
                    drop_last=False)

    return test, test_dl

def metrics_func(args, p, t5_tok, sim_input, e_model, e_tok, tops = (1, 3, 5), k_ent = 10):
    with torch.no_grad():
        preds, labels = p
        preds = preds.log_softmax(dim = -1)
        res = {}

        #Handle case when there is only one item in the batch
        if isinstance(preds, tuple):
            preds = preds[0]
        
        preds = preds.to(get_embed_device(args))
        preds = preds.type(torch.float32)
        labels = labels.to(get_embed_device(args))
    
        # NLL
        nll = nn.NLLLoss(reduction = 'none')
        nll = nll(preds.transpose(2, 1), labels)
        res['nll'] = (nll * (labels != -100)).sum(dim = -1).detach().cpu().tolist()

        # Top Entropy
        top_probs = torch.topk(preds, k = k_ent).values
        norm_probs = torch.exp(top_probs)
        entropy = (top_probs * norm_probs).sum(dim = -1).sum(dim = -1)
        res[f'top_{k_ent}_ent'] = (-entropy).detach().cpu().tolist()

        # Perplexity
        weights = torch.ones_like(preds[0, 0, :].squeeze())
        weights[[1,2,0]] = 0

        loss = nn.CrossEntropyLoss(reduction = 'none', weight = weights)
        loss = loss(preds.transpose(2, 1), target = labels) * (labels != -100)
        # perp = torch.exp(loss.sum(dim = -1))
        log_perp = loss.sum(dim = -1)
        res['log_perp'] = log_perp.detach().cpu().tolist()  

        #Top-K and K-Similarity

        gen, labels_sim, input_ids = sim_input
        
        top_k_gens  = gen.reshape((input_ids.shape[0], max(tops), gen.shape[-1]))[:, :max(tops), :].squeeze().reshape((input_ids.shape[0] * max(tops), gen.shape[-1]))
        top_k_texts = t5_tok.batch_decode(top_k_gens, skip_special_tokens = False)
        top_k_texts = [s.replace('<pad>', '') for s in top_k_texts]
        top_k_texts = [top_k_texts[i*max(tops):(i+1) * max(tops)] for i in range(len(top_k_texts)//max(tops))]
        
        ground   = t5_tok.batch_decode(labels, skip_special_tokens = False)

        ground = [re.sub(r'\</s\>|\<pad\>', '', g) for g in ground]
        ground = [[slot.strip() for slot in re.split(r'\<extra_id_\d+\>', g) if slot.strip() != ''] for g in ground]
        top_k_preds = []

        for i, samp_preds in enumerate(top_k_texts):
            samp_preds = [re.sub(r'\<pad\>', '', samp_pred) for samp_pred in samp_preds]
            samp_preds = [[slot.strip() for slot in re.split(r'\<extra_id_\d+\>|\</s\>', s) if slot.strip() != ''] for s in samp_preds]
            top_k_preds.append(samp_preds)
        
        
        dec_texts = t5_tok.batch_decode(input_ids, skip_special_tokens = False)
        dec_texts = [s.replace('<pad>', '') for s in dec_texts]
                
        for top in tops:
            sub_preds = [preds[:top] for preds in top_k_preds]
            sub_ground = ground

            top_k = top_k_text(sub_preds, sub_ground)
            res[f'acc_{top}'] = top_k
        
        sims = context_sim(top_k_preds, 
                          ground, dec_texts, 
                          e_model, e_tok, 
                          args['sim_type'],
                          tops = tops,
                          device = get_embed_device(args))
        
        for sim_top, sim in sims.items():
            res[f'top_{sim_top}_sim'] = sim
    
    return res

def eval_model(args, model, tok, e_model, e_tok, test_dl, logits_processor = None):
    res_dict = defaultdict(lambda: [])        
    
    i = 0
    for samp in tqdm(test_dl):
        input_ids = samp['input_ids'].to(get_device(args))
        attention_mask = samp['attention_mask'].to(get_device(args))
        labels = samp['labels'].to(get_device(args))
        labels[labels == -100] = 0
            
        outputs = model(input_ids = input_ids, 
                        attention_mask = attention_mask,
                        labels = labels)
        
        gen     = model.generate(input_ids = input_ids,
                        max_new_tokens = labels.shape[1] * 2,
                        num_beams = 5, 
                        num_return_sequences = 5)
        test_p  = (outputs.logits,  labels)
        
        decoded_labels = tok.batch_decode(labels, skip_special_tokens = False)
        decoded_labels = [s.replace('<pad>', '') for s in decoded_labels]
            
        sim_input = (gen, decoded_labels, input_ids)
        res = metrics_func(args, test_p, tok, sim_input, e_model, e_tok)    
        
        # Top K and Generations
        top_gens = gen.view((input_ids.shape[0], 5, gen.shape[-1]))[:,0,:]
        
        for k,v in res.items():
            res_dict[k] += v
    
        res_dict['pred']  += tok.batch_decode(top_gens, skip_special_tokens = False)
        res_dict['input'] += tok.batch_decode(input_ids, skip_special_tokens = False)
        res_dict['gold']  += tok.batch_decode(labels, skip_special_tokens = False)

        del input_ids
        del attention_mask
        del labels

    res_df = pd.DataFrame(res_dict)
    res_df['input'] = res_df['input'].apply(lambda l: l).str.replace('<pad>', '', regex = False)
    res_df['gold'] = res_df['gold'].apply(lambda l: l).str.replace('<pad>', '', regex = False)


    return res_df
    

if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
   
    model, tok = load_model(args)
    print(f'Loaded model and tokenizer from: {args["model_checkpoint"]}')

    e_model, e_tok = load_embed_model(args)
    print(f'Loaded embedding model and tokenizer from {args["embed_checkpoint"]}')

    data, test_dl = setup_data(args, model, tok)
    print(f'Loaded data from {args["test_fp"]}')

    eval_res = eval_model(args, model, tok, e_model, e_tok, test_dl)
    print(f'Finished evaluating model')

    eval_res['id'] = data['id']
    # Save predictions
    save_path = f'{args["result_dir"]}/{args["model_checkpoint"].split("/")[-1]}/internal/'
    if 'regional' in args['test_fp']:
        save_path += 'regional/'
    if 'trans' in args['test_fp']:
        save_path += 'trans/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    eval_res.to_csv(f'{save_path}/preds.csv', index = None)

    avg_dict = {k: sum(val for val in v if not np.isinf(val))/len(v) for k,v in eval_res.items() if k not in ('pred', 'input', 'gold', 'id') and len(v) > 0}
    avg_res  = pd.DataFrame(avg_dict, index = [0])
    avg_res.to_csv(f'{save_path}/metrics.csv', index = None)

    with open(f'{save_path}/config.json', 'w') as fp:
        args['time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(args, fp)

    print(f'Results saved to {save_path}')
    

    

    
