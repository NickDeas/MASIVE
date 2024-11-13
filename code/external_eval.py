import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas()

import os
import json
from collections import defaultdict
import argparse
import time

import transformers
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset

from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support

import nltk
from nltk.stem import WordNetLemmatizer 
import re
from numpy.linalg import norm

import random
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import itertools
import torch.nn.functional as F

from metrics import *

# Set Seeds
SEED = 13
random.seed(SEED) # Python
np.random.seed(SEED) # Numpy
torch.manual_seed(SEED) # Torch
transformers.set_seed(SEED) # Transformers

lemmatizer = WordNetLemmatizer()

LANG_DICT   = {'goemo': 'en',
               'goemo_full': 'en',
               'emoevent_en': 'en',
               'emoevent_es': 'es'}

LABELS_DICT = {'goemo': ['angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'nothing'],
               'goemo_full': ['admiration', 'amused', 'angry', 'annoyed', 'approving', 'caring', 'confused', 'curious', 'desire', 'disappointed', 'disapproval', 'disgusted', 'embarassed', 'excited', 'afraid', 'grateful', 'grief', 'happy', 'love', 'nervous', 'optimistic', 'proud', 'realized', 'relieved', 'remorseful', 'sad', 'surprised', 'nothing'],
               'emoevent_en': ['angry', 'disgusted', 'scared', 'happy', 'sad', 'surprised', 'nothing'],
               'emoevent_es': ['enojado', 'desagradado', 'asustado', 'feliz', 'triste', 'sorprendido', 'nada']
              }

FP_DICT = {'goemo': 'goemo_ekman_test.csv',
           'goemo_full': 'goemo_full_test.csv',
           'emoevent_en': 'emo_event_en_test.csv',
           'emoevent_es': 'emo_event_es_test.csv'}

def parse_args():
    
    parser = argparse.ArgumentParser(
        prog = 'MASIVE External Eval',
        description = 'Evaluate a T5/mT5 model on prior emotion detection benchmarks.'
    )
    
    parser.add_argument('-cf', '--config',
                        type = str)

    parser.add_argument('-mc', '--model-checkpoint',
                        type = str)
    
    parser.add_argument('-ec', '--embed-checkpoint',
                        type = str,
                        default = 'bert-base-multilingual-uncased')

    parser.add_argument('-t', '--test-dir',
                        type = str)
    
    parser.add_argument('-e', '--eval-datas',
                        type = str,
                        nargs='+',
                        default = ['goemo', 'emoevent_en'])
    
    parser.add_argument('-i', '--input-col',
                        type = str,
                        default = 'text')
    
    parser.add_argument('-tc', '--label-col',
                        type = str,
                        default = 'label')
    
    parser.add_argument('-st', '--sim-type',
                        type = str,
                        choices = ['min', 'max', 'avg'],
                        default = 'max')
    
    parser.add_argument('-r', '--result-dir',
                        type = str)

    parser.add_argument('-d', '--device',
                        type = int)
    
    parser.add_argument('-ed', '--embed-device',
                        type = int,
                        default = -1)
    
    args = vars(parser.parse_args())
    
    if args['config'] is not None:
        with open(args['config'], 'r') as f:
            args.update(json.load(f))
    
    return args

def get_device(args):
    return torch.device(f'cuda:{args["device"]}')

def get_embed_device(args):
    if args['embed_device'] == -1:
        return get_device(args)
    else:
        return torch.device(f'cuda:{args["device"]}')

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

def tokenize_sample_data(data, src, trg):
    input_feature = tok(data[src], truncation=True, max_length = 384, padding = 'longest')
    label         = tok(data[trg], truncation=True, padding = 'longest')

    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }

def tok_data(src, trg):
    return lambda data: tokenize_sample_data(data, src, trg)

def format_df(args, df, ds_labels, lang):
    data = df.copy()
    data = data[['id', args['input_col'], args['label_col']]]
    data['label_text'] = data[args['label_col']].apply(lambda labels: [ds_labels[int(label)] for label in labels.split(',')]) 

    #English
    if lang == 'en':
        data['input'] = data[args['input_col']] + ' I feel <extra_id_0>'
    #Spanish
    elif lang == 'es':
        data['input'] = data[args['input_col']] + ' Me siento <extra_id_0>'

    data['ground'] = ' <extra_id_0> ' + data['label_text'].apply(lambda lab_l: lab_l[0])
    
    return data

def setup_data(args, data_fp, ds_labels, lang = 'en'):
    
    test = pd.read_csv(args['test_dir'] + '/' + data_fp)
    test = test.astype(str)
    
    test = format_df(args, test, ds_labels, lang)
    
    return test

def make_pred(args, model, tok, text, labels_enc, loss, ds_labels, lang = 'en'):
    #English
    if lang == 'en':
        input_txts = [text + ' I feel <extra_id_0>'] * len(ds_labels)
    #Spanish
    elif lang == 'es':
        input_txts = [text + ' Me siento <extra_id_0>'] * len(ds_labels)
    
    input_enc = tok(input_txts, return_tensors = 'pt').to(get_device(args))

    
    # Classification Predictions
    preds = model(**input_enc, 
                  labels = labels_enc
                 )
    
    losses   = loss(preds.logits.permute(0, 2, 1), labels_enc)

    probs    = (-losses.sum(dim = -1)).softmax(dim = -1)
    
    # Generated Raw Predictions
    gens = model.generate(input_enc['input_ids'][0:1],
                         num_beams = 5,
                         num_return_sequences = 5,
                         max_new_tokens = 16)
    
    return input_enc['input_ids'], preds.logits, probs.detach().cpu().tolist(), gens

def metrics_func(args, p, t5_tok, sim_input, e_model, e_tok, tops = (1, 3, 5), k_ent = 10):
    with torch.no_grad():
        preds, labels = p
        preds = preds.log_softmax(dim = -1)
        res = {}

        #Handle case when there is only one item in the batch
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = preds.to(get_device(args))
        preds = preds.type(torch.float32)
        labels = labels.to(get_device(args))
    
        # --- NLL ---
        nll = nn.NLLLoss(reduction = 'none')
        nll = nll(preds.transpose(2, 1), labels)
        res['nll'] = (nll * (labels != -100)).sum(dim = -1).detach().cpu().tolist()

        # --- Top Entropy ---
        top_probs = torch.topk(preds, k = k_ent).values
        norm_probs = torch.exp(top_probs)
        entropy = (top_probs * norm_probs).sum(dim = -1).sum(dim = -1)
        res[f'top_{k_ent}_ent'] = (-entropy).detach().cpu().tolist()

        # --- Perplexity ---
        weights = torch.ones_like(preds[0, 0, :].squeeze())
        weights[t5_tok.all_special_ids] = 0

        loss = nn.CrossEntropyLoss(reduction = 'none', weight = weights)
        loss = loss(preds.transpose(2, 1), target = labels) * (labels != -100)
        perp = torch.exp(loss.sum(dim = -1))
        res['perp'] = perp.detach().cpu().tolist()     

        
        # --- Top-K and K-Similarity ---
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

def eval_model(args, model, tok, e_model, e_tok, test, ds_labels, lang = 'en'):
    
    res_dict = defaultdict(lambda: [])
    
    pos_labs = tok([' <extra_id_0> ' + emo for i, emo in enumerate(ds_labels)], return_tensors = 'pt', padding = True).to(get_device(args))
    pos_labs = pos_labs['input_ids']
    
    loss = nn.CrossEntropyLoss(reduction = 'none', ignore_index = -100)
    
    for i, row in enumerate(tqdm(test.to_dict(orient = 'records'))):
        
        if len(str(row[args['label_col']]).split(',')) > 1:
            continue
        
        text = str(row[args['input_col']])
        label_int = int(row[args['label_col']])
        
        input_ids, preds, probs, gens = make_pred(args, model, tok, text, pos_labs, loss, ds_labels, lang)
        
        decoded_labels = tok.batch_decode(pos_labs[label_int])
        
        test_p = (preds[label_int].unsqueeze(0), pos_labs[label_int].unsqueeze(0))
        sim_input = (gens.unsqueeze(0), decoded_labels, input_ids[0].unsqueeze(0))
        res = metrics_func(args, test_p, tok, sim_input, e_model, e_tok)    
        
        gen_texts = tok.batch_decode(gens)
        
        res_dict['probs'].append(probs)
        res_dict['top5'].append(gen_texts)
                                          
        for k,v in res.items():
            res_dict[k] += v
            
        res_dict['id'].append(row['id'])
        res_dict['pred'].append(tok.batch_decode(gens[0]))
        res_dict['input'].append(row[args['input_col']])
        res_dict['gold'].append(row['label_text'][0])
        
    res_df = pd.DataFrame(res_dict)
    res_df['input'] = res_df['input'].apply(lambda l: l).str.replace('<pad>', '', regex = False)
    
    return res_df

def create_vec(label, ds_labels):
    lab_vec = np.zeros(len(ds_labels))
    for lab in label.split(','):
        lab_vec[int(lab)] = 1
    return lab_vec

def create_pred_vec(pred, ds_labels):
    lab_vec = np.zeros(len(ds_labels))
    lab_vec[ds_labels.index(pred)] = 1
    return lab_vec

def class_eval(res_df, ds_labels):
    
    res_df['pred_label'] = res_df['probs'].progress_apply(lambda pl: ds_labels[pl.index(max(pl))])
    res_df['pred_int']   = res_df['probs'].progress_apply(lambda pl: pl.index(max(pl)))
    
    labels = res_df['gold'].tolist()
    preds  = res_df['pred_label'].tolist()
    
    precs, recs, fscores, supports = precision_recall_fscore_support(labels, preds, average = None, labels = ds_labels, zero_division = 0)
    acc = accuracy_score(labels, preds)
        
    res_dict = {'acc': acc}
    for i, label in enumerate(ds_labels):
        res_dict[f'{label}_prec']    = precs[i]
        res_dict[f'{label}_rec']     = recs[i]
        res_dict[f'{label}_f1']      = fscores[i]
        res_dict[f'{label}_support'] = supports[i]
        
    res_dict[f'micro_prec']  = sum([p*s for p,s in zip(precs, supports)])/sum(supports)
    res_dict[f'micro_rec']   = sum([p*s for p,s in zip(recs, supports)])/sum(supports)
    res_dict[f'micro_f1']    = sum([p*s for p,s in zip(fscores, supports)])/sum(supports)
    res_dict[f'macro_prec']  = sum(precs)/len(precs)
    res_dict[f'macro_rec']   = sum(recs)/len(recs)
    res_dict[f'macro_f1']    = sum(fscores)/len(fscores)
    
    return res_dict

if __name__ == '__main__':
    
    args = parse_args()
    print('Parsed arguments: \n' + "\n".join(['\t' + str(name) + ': ' + str(val) for name, val in args.items()]))
   
    model, tok = load_model(args)
    print(f'Loaded model and tokenizer from: {args["model_checkpoint"]}')

    e_model, e_tok = load_embed_model(args)
    print(f'Loaded embedding model and tokenizer from {args["embed_checkpoint"]}')

    for i, data_name in enumerate(args['eval_datas']):
        lang = LANG_DICT[data_name]
        ds_labels = LABELS_DICT[data_name]
        
        data = setup_data(args, FP_DICT[data_name], ds_labels, lang = lang)
        print(f'\tLoaded data from {args["test_dir"] + FP_DICT[data_name]} ({i} of {len(args["eval_datas"])}')

        eval_res = eval_model(args, model, tok, e_model, e_tok, data, ds_labels, lang = lang)
        print(f'\tFinished evaluating model')

        eval_res['id'] = data['id']
        
        # Save predictions
        save_path = f'{args["result_dir"]}/{args["model_checkpoint"].split("/")[-1]}/external/{data_name}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        eval_res.to_csv(f'{save_path}/preds.csv', index = None)
        
        avg_dict = {k: sum(val for val in v if not np.isinf(val))/len(v) for k,v in eval_res.items() if k not in ('pred', 'input', 'gold', 'id', 'probs', 'top5') and len(v) > 0}
        class_dict = class_eval(eval_res, ds_labels)
        avg_dict.update(class_dict)
        
        avg_res  = pd.DataFrame(avg_dict, index = [0])
        avg_res.to_csv(f'{save_path}/metrics.csv', index = None)

        with open(f'{save_path}/config.json', 'w') as fp:
            args['time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            json.dump(args, fp)

        print(f'\tResults saved to {save_path}')