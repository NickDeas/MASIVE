import torch
from torch.nn import functional as F
import numpy as np
from numpy.linalg import norm

import re
from collections import defaultdict


# --- Regex Expressions ---
# Compiled pattern to identify slots in the sentence
input_pattern = re.compile(r"<extra_id_\d+>|<MASK>")
# Compiled pattern to identify gold emotions in targets
gold_pattern = re.compile(r"(?:<extra_id_\d+>|<MASK>)\s*(.*?)\s*(?=<extra_id_\d+>|<MASK>|</s>|$)")
pattern = re.compile(r'(?:<extra_id_\d+>|<MASK>)\s*(.*?)\s*(?=<extra_id_\d+>|<MASK>|$)')

CONTEXT_SIZE = 100

def cosine_sim(embed_1, embed_2):
    if embed_1 is None or embed_2 is None:
        return 0
    cos_sim = F.cosine_similarity(embed_1, embed_2, dim = 0)
    return cos_sim.cpu().item()

def top_k_text(top_preds, ground_truth):
    top_ks = []
    for samp_preds, samp_ground in zip(top_preds, ground_truth):
        correct, total = 0, 0
        for i, label in enumerate(samp_ground):
            pred_slots = [samp_pred[i] for samp_pred in samp_preds if len(samp_pred) > i]
            if label in pred_slots:
                correct += 1
        
        acc = correct/len(samp_ground)
        top_ks.append(acc)

    return top_ks


def find_in_tensor(full_tensor, subtensor, start = 0):
    
    windows = torch.arange(subtensor.shape[0]).repeat(full_tensor.shape[0] - subtensor.shape[0] + 1, 1)
    windows = torch.arange(full_tensor.shape[0] - subtensor.shape[0] + 1).unsqueeze(1) + windows

    predicted_idx = (full_tensor[windows[start:]] == subtensor).sum(-1).nonzero()
    if predicted_idx.size(0) == 0:
        return None
    
    return start + predicted_idx[0][0].item()


def get_term_embeds(full_text, term_set, 
               e_model, e_tok,
               num_embeds = None,
               device = torch.device('cpu')):
    
    # Tokenize context
    full_tokenized = e_tok.batch_encode_plus(
        [full_text],
        padding=False,
        truncation=False,
        return_tensors='pt',
        add_special_tokens=False
    )
    
    full_input_ids      = full_tokenized['input_ids'][0]
    full_attention_mask = full_tokenized['attention_mask'][0]

    # Truncate if more predictions than give number of embeds
    if num_embeds is not None:
        term_set = term_set[:num_embeds]
    
    term_idxs = []
    start = 0
    # Iterate over terms
    for term in term_set:
        term_tokenized = e_tok.batch_encode_plus(
            [term],
            padding=False,
            truncation=False,
            return_tensors='pt',
            add_special_tokens=False
        )
        
        term_input_ids = term_tokenized['input_ids'][0]
        
        # Iteratively find idxs of terms in filled context
        if start + term_input_ids.shape[0] < full_input_ids.shape[0]:
            term_idx = find_in_tensor(full_input_ids, term_input_ids, start = start)
            if term_idx is None:
                term_idxs.append(None)
            else:
                term_idxs.append(term_idx)
                start = term_idx + term_input_ids.shape[0]

    # Extract contextual embeddings
    term_embeds = []
    for term_idx in term_idxs:
        if term_idx is not None:
            context_min_idx = max(term_idx - CONTEXT_SIZE, 0)
            context_max_idx = min(term_idx + CONTEXT_SIZE, full_input_ids.shape[0])

            with torch.no_grad():
                all_embeds = e_model(full_input_ids[context_min_idx: context_max_idx].unsqueeze(0).to(device), attention_mask=full_attention_mask[context_min_idx: context_max_idx].unsqueeze(0).to(device))

            all_embeds = all_embeds.last_hidden_state
            term_embed  = all_embeds[0][term_idx - context_min_idx]
            term_embeds.append(term_embed)
        else:
            term_embeds.append(None)
    
    # Fill in missing values with None
    if num_embeds is not None:
        term_embeds = term_embeds + [None] * (num_embeds - len(term_embeds))
    
    return term_embeds

def fill_context(context, terms):
    terms_ = terms[::]
    def replace_with_emotions(match):
            # Pop the first emotion from the list to replace the current slot
            if terms_:
                return (" " if match.start() > 0 and context[match.start() - 1] not in [' ', '\n'] else "") + terms_.pop(0)
            else:
                return ""

    # Replace slots in the sentence with emotions
    filled_context = re.sub(input_pattern, replace_with_emotions, context)

    return filled_context

def calc_avg_sim(embeds1, embeds2):
    sims = []
    for embed1, embed2 in zip(embeds1, embeds2):
        sim = cosine_sim(embed1, embed2)
        sims.append(sim)
    return np.average(sims)

def single_context_sim(top_preds, gold_labels, context,
                       e_model, e_tok,
                       tops = (1,3,5),
                       agg_type = 'max',
                       device = torch.device('cpu')):
    

    # Pull out gold embeddings
    num_gold     = len(gold_labels)
    gold_context = fill_context(context, gold_labels)

    
    gold_embeds  = get_term_embeds(gold_context, gold_labels, e_model, e_tok, device = device)
    
    topk_sims = []
    # Iterate over ranked predictions
    for j, top_pred in enumerate(top_preds):
        
        # Get embeds for this prediction set
        full_text = fill_context(context, top_pred)
        pred_embeds = get_term_embeds(full_text, top_pred, 
                                      e_model, e_tok, 
                                      num_embeds = num_gold, 
                                      device = device)

        sim = calc_avg_sim(gold_embeds, pred_embeds)
        topk_sims.append(sim)
    
    top_sim_dict = {}
    for top in tops:
        if agg_type == 'min':
            top_sim_dict[top] = np.min(topk_sims[:top])
        elif agg_type == 'max':
            top_sim_dict[top] = np.max(topk_sims[:top])
        elif agg_type == 'avg':
            top_sim_dict[top] = np.average(topk_sims[:top])
            
    return top_sim_dict
        
    

def context_sim(top_preds, ground_truth, inputs, 
                e_model, e_tok, 
                agg_type = 'max', 
                tops = (1,3,5),
                device = torch.device('cpu')):
    '''
        Calculate the top-k similarity scores for a batch of predictions
        
        Parameters:
            -top_preds: List of Top-k predictions per sample. Should be of size batch_size x k x num_preds]. k should be equal to the maximum value in `tops`
            -ground_truth: List of labels per sample. Should be of size batch_size x num_labels
            -inputs: List of input contexts. Should be of size batch_size
            -agg_type: how to aggregate the resulting similarity scores across top-k predictions
            -tops: Different values of k to calculate top-k similarity for
    '''
     
    similarities = defaultdict(list)

    # Iterate over batch
    for topk, gold_emo_set, context in zip(top_preds, ground_truth, inputs):
        # Get topk-similarity scores for single sample
        single_sim = single_context_sim(topk, gold_emo_set, context, 
                                        e_model, e_tok,
                                        agg_type = agg_type, tops = tops, 
                                        device = device)
        # Update results
        for key, val in single_sim.items():
            similarities[key].append(val)
    
    return similarities
