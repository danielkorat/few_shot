from transformers import pipeline
from sys import stdout
import torch
import spacy

from patterns import PATTERNS, SCORING_PATTERNS

models_dict = {}

spacy_model = spacy.load('en_core_web_sm')

def get_fm_pipeline(model):
    if model in models_dict:
        fm_model = models_dict[model]
    else:
        print(f"\nLoading {model} fill-mask pipeline...\n")
        stdout.flush()
        fm_model = pipeline('fill-mask', model=model, framework="pt")
        models_dict[model] = fm_model
    return fm_model

def extract_aspects(text, tokens, pattern_names, scoring_patterns=None,
                                    top_k=10, thresh=-1, target=True, **kwargs):
    
    hparams = locals()
    for v in 'text', 'tokens', 'kwargs':
        hparams.pop(v)
    hparams.update(kwargs)
    fm_pipeline = get_fm_pipeline(kwargs['model_name'])

    preds, pred_bio = extract_candidate_aspects(fm_pipeline, text, tokens, pattern_names,
                                                top_k, thresh, target_flag=True, **kwargs)

    if scoring_patterns:
        #--------- aspect scoring --------
        
        pattern = SCORING_PATTERNS[scoring_patterns[0]]
        valid_preds=[]

        for pred in preds:
            #target_terms=['good', 'great', 'amazing', 'bad', 'awful', 'horrible']
            #target_terms=['positive', 'negative', 'neutral', 'ok', 'none']
            target_terms=['Yes', 'No']
            # add leading space to targets as mask predictions function needs
            target_terms = [' '+target for target in target_terms ]
            mask_preds = fill_mask_preds(fm_pipeline, text, target_terms, pattern, top_k, target_flag=True, sapect_token=pred)

            #if mask_preds[0]['score']>0.02:
            score_ratio = mask_preds[0]['score']/mask_preds[1]['score']
            
            if mask_preds[0]['token_str']==' Yes':
                valid_preds.append(pred)
            else:
                if score_ratio<1.5:
                    valid_preds.append(pred)

                
            #calc pred_bio again
        preds = valid_preds    
        pred_bio = generate_bio(tokens=tokens, preds=preds)    
    
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
        if pred =='pr':
            print("sdsd")
        idx = tokens.index(pred) 
        idx_all.append(idx)

    pred_bio = ['B-ASP' if i in idx_all else 'O' for i in range(len(tokens))]

    return pred_bio


def fill_mask_preds(fm_pipeline, text, target_terms, pattern, top_k, target_flag, sapect_token=None):
    delim = ' ' if text[-1] in ('.', '!', '?') else '. '
    pattern = pattern.replace('<mask>', f"{fm_pipeline.tokenizer.mask_token}")
    if sapect_token:
        pattern = pattern.replace('<aspect>', sapect_token)
    mask_preds = fm_pipeline(delim.join([text, pattern]), top_k=top_k,
                            targets=target_terms if target_flag else None)
    
    return(mask_preds)
