from transformers import pipeline
from sys import stdout
import spacy

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

def extract_aspects_from_sentence(PATTERNS, text, tokens, model_name, pattern_names, top_k=10, thresh=-1, target=True, **kwargs):
    hparams = locals()
    for v in 'text', 'tokens', 'kwargs':
        hparams.pop(v)
    hparams.update(kwargs)
    fm_pipeline = get_fm_pipeline(model_name)

    # Single patterns
    if len(pattern_names) == 1:
        pattern = PATTERNS[pattern_names[0]]
        mask_preds = fill_mask_preds(fm_pipeline, text, tokens, pattern, top_k, target)
        
    # Multiple patterns
    else:
        mask_preds_all = []
        for pattern_name in pattern_names:
            pattern = PATTERNS[pattern_name]
            mask_preds = fill_mask_preds(fm_pipeline, text, tokens, pattern, top_k, target)
            mask_preds_all.append(mask_preds)
        
        mask_preds = merge_mask_preds(mask_preds_all=mask_preds_all, strategy='union')

    preds = [pred['token_str'].lstrip() for pred in mask_preds]

    # validate - make sure token exists in sentece
    valid_preds, pred_bio = validate_pred_tokens(text, tokens, mask_preds, thresh = thresh)
    
    return preds, valid_preds, pred_bio, mask_preds, hparams


def validate_pred_tokens(text, tokens, mask_preds, thresh=-1):
    valid_preds, valid_idx = [], set()

    for pred in mask_preds:
        pred_token, score = pred['token_str'].lstrip(), pred['score']

        if score > thresh:
            try:
                idx = tokens.index(pred_token)                
                if (is_noun_token(text, pred_token)):
                    valid_idx.add(idx)
                    valid_preds.append((pred_token, f"{score:.3f}"))
            except ValueError:
                pass

    pred_bio = ['B-ASP' if i in valid_idx else 'O' for i in range(len(tokens))]

    return valid_preds, pred_bio

def is_noun_token(text, pred_token):

    nouns = [ent.text for ent in spacy_model(text) if ent.pos_ == 'NOUN']
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


def fill_mask_preds(fm_pipeline, text, tokens, pattern, top_k, target):
    delim = ' ' if text[-1] in ('.', '!', '?') else '. '
    pattern = pattern.replace('<mask>', f"{fm_pipeline.tokenizer.mask_token}")
    mask_preds = fm_pipeline(delim.join([text, pattern]), top_k=top_k,
                            target=tokens if target else None)
        
    return(mask_preds)