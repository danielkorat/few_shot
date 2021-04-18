from transformers import pipeline
from sys import stdout
import torch
import spacy

from patterns import PATTERNS, SCORING_PATTERNS

MODELS_DICT = {}

spacy_model = spacy.load('en_core_web_sm')

def get_fm_pipeline(model_name):
    # If fine-tuned, prepend dir name and do not keep in memory
    if 'rest_' in model_name or 'lap_' in model_name:
        fm_pipeline = pipeline('fill-mask', model=f"models/{model_name}", framework="pt")

    # Otherwise, load pipeline and keep in memory
    elif model_name in MODELS_DICT:
        fm_pipeline = MODELS_DICT[model_name]
    else:
        print(f"\nLoading {model_name} fill-mask pipeline...\n")
        stdout.flush()
        fm_pipeline = pipeline('fill-mask', model=model_name, framework="pt")
        MODELS_DICT[model_name] = fm_pipeline

    return fm_pipeline

def extract_aspects(fm_pipeline, text, tokens, pattern_names, model_name, scoring_patterns=None,
                            top_k=10, thresh=-1, target=True, **kwargs):
    
    hparams = locals()
    for v in 'text', 'tokens', 'kwargs':
        hparams.pop(v)
    hparams.update(kwargs)
    
    preds, pred_bio = extract_candidate_aspects(fm_pipeline, text, tokens, pattern_names,
                                                top_k, thresh, target_flag=True, **kwargs)

    if scoring_patterns:
        #--------- aspect scoring --------
        
        pattern = SCORING_PATTERNS[scoring_patterns[0]]
        for pred in preds:
            target_terms=['good', 'great', 'amazing', 'bad', 'awful', 'horrible']
            #mask_preds = fill_mask_preds(fm_pipeline, text, target_terms, pattern, top_k, target_flag=True, aspect_token=pred)
            mask_preds = fill_mask_preds(fm_pipeline, text, target_terms, pattern, top_k=1000, target_flag=False, aspect_token=pred)

            print ("mask preds: ", mask_preds)
    
    return preds, pred_bio, hparams


def extract_candidate_aspects(fm_pipeline, text, tokens, pattern_names,
                                    top_k=10, thresh=-1, **kwargs):

    nouns = [ent.text for ent in spacy_model(text) if ent.pos_ == 'NOUN']

    # Single patterns
    if len(pattern_names) == 1:
        pattern = PATTERNS[pattern_names[0]]
        mask_preds = fill_mask_preds(fm_pipeline, text, tokens, pattern, top_k, target_flag=False)
        mask_preds = validate_pred_tokens(text, tokens, mask_preds, nouns, thresh = thresh)
        
    # Multiple patterns
    else:
        mask_preds_all = []
        for pattern_name in pattern_names:
            pattern = PATTERNS[pattern_name]
            mask_preds = fill_mask_preds(fm_pipeline, text, tokens, pattern, top_k, target_flag=False)
            # validate - make sure token exists in sentece and are nouns
            mask_preds = validate_pred_tokens(text, tokens, mask_preds, nouns, thresh = thresh)
            mask_preds_all.append(mask_preds)
        
        
        mask_preds = merge_mask_preds(mask_preds_all=mask_preds_all, strategy='union')
        # if mask_preds:
        #     mask_preds = score_aspects(fm_pipeline, text, mask_preds, mask_preds_all,PATTERNS, pattern_names)

    preds = [pred['token_str'].lstrip() for pred in mask_preds]
    pred_bio = generate_bio(tokens=tokens, preds=preds)

    return preds, pred_bio


def fill_mask_preds(fm_pipeline, text, target_terms, pattern, top_k, target_flag, aspect_token=None):
    delim = ' ' if text[-1] in ('.', '!', '?') else '. '
    pattern = pattern.replace('<mask>', f"{fm_pipeline.tokenizer.mask_token}")
    if aspect_token:
        pattern = pattern.replace('<aspect>', aspect_token)
    mask_preds = fm_pipeline(delim.join([text, pattern]), top_k=top_k,
                            targets=target_terms if target_flag else None)
    return mask_preds
    

def score_aspects(fm_pipeline, text, preds, preds_all, PATTERNS, pattern_names):
    
    asp_cands = [pred['token_str'].lstrip() for pred in preds]
    delim = ' ' if text[-1] in ('.', '!', '?') else '. '
    softmax = torch.nn.Softmax(dim=1)

    for pattern_name in pattern_names:
        pattern = PATTERNS[pattern_name]
        pattern = pattern.replace('<mask>', f"{fm_pipeline.tokenizer.mask_token}")
        mask_preds = fm_pipeline(delim.join([text, pattern]), targets=asp_cands, top_k=None)

        scores = [mask_pred['score'] for mask_pred in mask_preds]
      
        scores = torch.FloatTensor([scores])
        # mx=torch.max(scores,1)
        # scores = scores/mx.values
        scores_softmax = softmax(scores)
        
        scores_softmax = scores_softmax.tolist()
        i=0
        for mask_pred in mask_preds:
            mask_pred['score'] = scores_softmax[0][i]
            i=i+1

    return mask_preds                               

def validate_pred_tokens(text, tokens, mask_preds,nouns , thresh=-1):
    valid_preds, valid_idx = [], set()

    for pred in mask_preds:
        pred_token, score = pred['token_str'].lstrip(), pred['score']

        if score > thresh:
            try:
                idx = tokens.index(pred_token)                
                if (is_noun_token(text, pred_token, nouns)):
                    valid_idx.add(idx)
                    #valid_preds.append((pred_token, f"{score:.3f}"))
                    valid_preds.append(pred)
            except ValueError:
                pass

    return valid_preds


def is_noun_token(text, pred_token, nouns):

    if pred_token in nouns:
        return True
    else:
        return False

def merge_mask_preds(mask_preds_all, strategy='union'):
    
    if strategy=='union':
        unified_preds = mask_preds_all[0]
        for mask_preds in mask_preds_all[1:]:
            for pred in mask_preds:
                pred_tokens = pred['token']
                if pred_tokens not in [pred['token'] for pred in unified_preds]:
                    unified_preds.append(pred)
    
    return unified_preds


def generate_bio(tokens, preds):     
    idx_all =[]
    for pred in preds:
        idx = tokens.index(pred) 
        idx_all.append(idx)

    pred_bio = ['B-ASP' if i in idx_all else 'O' for i in range(len(tokens))]

    return pred_bio
