from transformers import pipeline
from seqeval.metrics import f1_score, precision_score, recall_score,\
                            performance_measure
import csv, json, pickle, spacy, requests, logging
import pandas as pd
from urllib.request import urlopen
from sys import stdout
import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np

logging.disable(logging.WARNING)

from tqdm import tqdm as tq
def tqdm(iter, **kwargs):
    return tq(list(iter), kwargs, position=0, leave=True, file=stdout)

DATA_DIR = 'https://raw.githubusercontent.com/IntelLabs/nlp-architect/libert/nlp_architect/models/libert/data/'
CSV_DATA_DIR = DATA_DIR + 'csv/spacy/domains_all/'
JSON_DATA_DIR = DATA_DIR + 'Dai2019/semeval14/'

# Mask Patterns
P1 = "So, the <mask> is the interesting aspect."
P2 = "So, the interesting aspect is <mask>."
P3 = "So, the <mask> are the interesting aspect."
P4 = "So, this is my opinion on <mask>."
P5 = "So, my review focuses on the <mask>."

models_dict = {}
# domain_ds = {'res': res_ds, 'lap': lap_ds}

# nlp = spacy.load("en_core_web_sm", \
#              disable=["parser", "ner", "entity_linker", "textcat",
#                       "entity_ruler", "sentencizer",
#                       "merge_noun_chunks", "merge_entities",
#                       "merge_subtokens"])
    
# def lemma(nlp, token):
#     return nlp(token)[0].lemma_.lower()

def load_dataset(csv_url, json_url, multi_token=False):
    print(f"Loading dataset from '{csv_url}' and '{json_url}'...")
    json_rows = requests.get(JSON_DATA_DIR + json_url).text.splitlines()

    row_break = "\r\n_,_,_,_,_,_,_\r\n_,_,_,_,_,_,_\r\n"
    csv_text = requests.get(CSV_DATA_DIR + csv_url).text
    csv_text = csv_text[csv_text.index('\n') + 1:] # skip csv header
    csv_sents = csv_text.split(row_break)[:len(json_rows)] # clip train data
    ds, aspects = [], []

    for json_row, csv_sent in zip(json_rows, csv_sents):
        json_obj = json.loads(json_row)
        text = json_obj['text']
        tokens, labels, aspects = [], [], []
        for token, label, *_ in csv.reader(csv_sent.splitlines()):
            if label == 'I-ASP':
                aspects = None
                break
            tokens.append(token)
            if label == 'B-ASP':
                aspects.append(token)
                labels.append(label)
            else:
                labels.append('O')
        if aspects == None and not multi_token:
            continue
        if multi_token and 'terms' in json_obj:
            aspects = [t['term'] for t in json_obj['terms']]
        ds.append((text, tokens, labels, aspects))
    return ds

def load_all_datasets(verbose=False, train_size=100):
    res_ds = load_dataset('restaurants.csv', "restaurants/restaurants_train_sents.json")
    lap_ds = load_dataset('laptops.csv', "laptops/laptops_train_sents.json")

    if verbose:
        print(f'RESTAURANTS (size={len(res_ds)}):\n')
        for ex in res_ds[:5]:
            print(ex[0], ex[3])

        print(f'\n\nLAPTOPS (size={len(lap_ds)}):\n')
        for ex in lap_ds[:5]:
            print(ex[0], ex[3])

    res_train, res_test = res_ds[:train_size], res_ds[train_size:]
    lap_train, lap_test = lap_ds[:train_size], lap_ds[train_size:]
    print(f"Restaurants size (train/test): {len(res_train)}/{len(res_test)}")
    print(f"Laptops size (train/test): {len(lap_train)}/{len(lap_test)}")
    return (res_train, res_test), (lap_train, lap_test)

def get_fm_pipeline(model, device=0):
    if model in models_dict:
        fm_model = models_dict[model]
    else:
        print(f"\nLoading {model} fill-mask pipeline...\n")
        stdout.flush()
        fm_model = pipeline('fill-mask', model=model, framework="pt", device=device)
        models_dict[model] = fm_model
    return fm_model

def run_example(text, tokens, model, pattern, top_k=10, thresh=-1, target=True, device=3):
    hparams = locals()
    hparams.pop('text')
    hparams.pop('tokens')

    delim = ' ' if text[-1] in ('.', '!', '?') else '. '
    
    fm_pipeline = get_fm_pipeline(model)
    pattern = pattern.replace('<mask>', f"{fm_pipeline.tokenizer.mask_token}")
    preds_meta = fm_pipeline(delim.join([text, pattern]), top_k=top_k,
                         target=tokens if target else None)
    preds, valid_preds, valid_idx = [], [], set()

    for pred in preds_meta:
        pred_token, score = pred['token_str'].lstrip(), pred['score']
        preds.append(pred_token)

        if score > thresh:
            try:
                idx = tokens.index(pred_token)
                valid_idx.add(idx)
                valid_preds.append((pred_token, f"{score:.3f}"))
            except ValueError:
                pass

    pred_bio = ['B-ASP' if i in valid_idx else 'O' for i in range(len(tokens))]
    return preds, valid_preds, pred_bio, preds_meta, hparams

def run_ds_examples(ds, model, **kwargs):
    print(f"Pattern: {kwargs['pattern']}\n")
    for i, (text, tokens, gold_bio, aspects) in tqdm(enumerate(ds)):
        preds, valid_preds, pred_bio, _, _ = run_example(model=model, text=text, tokens=tokens, **kwargs)
        print(i, text)
        print(tokens)
        print(f'gold: {aspects}\ngold_bio: {gold_bio}\nvalid_preds: {valid_preds}\npreds: {preds}\npred_bio: {pred_bio}\n')
    
def eval_domain(domain, limit=None, **kwargs):
    all_preds_bio, all_preds, all_preds_meta, all_gold_bio = [], [], [], []
    for text, tokens, gold_bio, aspects in domain_ds[domain][:limit]:
        preds, _, pred_bio, preds_meta, hparams = run_example(text=text, tokens=tokens, **kwargs)
        all_preds.append(preds)
        all_preds_bio.append(pred_bio)
        all_preds_meta.append(preds_meta)
        all_gold_bio.append(gold_bio)

    with open(f'{domain}.pkl', 'wb') as f:
        pickle.dump((all_preds, all_preds_meta), f)

    return {'metrics': metrics(all_gold_bio, all_preds_bio, domain, **kwargs), 'hparams': hparams}

def metrics(gold, preds, domain, verbose=False, **kwargs):
    F, P, R, conf = (f(gold, preds) for f in (f1_score, precision_score,\
                     recall_score, performance_measure))
    if verbose:
        print(f'{domain}')
        print(f'F1: {F:.3f}, P: {P:.3f}, R: {R:.3f}, {conf}')
    return {'F1': F, 'Precision': P, 'Recall': R}

def post_process(domain, limit=None, thresh=-1, **kwargs):
    with open(f'{domain}.pkl', 'rb') as f:
        _, all_preds_meta = pickle.load(f)

    nlp = spacy.load("en_core_web_sm", \
                     disable=["parser", "ner", "entity_linker", "textcat",
                              "entity_ruler", "sentencizer",
                              "merge_noun_chunks", "merge_entities",
                              "merge_subtokens"])
    
    # TODO: lemmatize sentences, not token lists

    all_preds_bio, all_gold_bio = [], []
    for (text, tokens, gold_bio, aspects), preds_meta in \
        zip(domain_ds[domain][:limit], all_preds_meta[:limit]):

        pred_lems, valid_preds, token_lems = [], [], []
        for t in tokens:
            toks = list(nlp(t))
            token_lems.append(toks[0].lemma_ if toks else [''])

        valid_idx = set()
        for pred in preds_meta:
            pred_token = pred['token_str'].lstrip() #'Ġ'
            score = pred['score']
            if score > thresh:
                pred_as_tokens = list(nlp(pred_token))
                if pred_as_tokens:
                    pred_lem = pred_as_tokens[0].lemma_
                    try:
                        valid_idx.add(token_lems.index(pred_lem))
                    except ValueError:
                        pass
        pred_bio = ['B-ASP' if i in valid_idx else 'O' for i in range(len(tokens))]
        all_preds_bio.append(pred_bio)
        all_gold_bio.append(gold_bio)
    return {'metrics': metrics(all_gold_bio, all_preds_bio, domain)}

def eval_all(**kwargs):    
    eval_res, post_res = {}, {}
    hparams = None
    for domain in tqdm(('res', 'lap')):
        for res, func in tqdm(zip((eval_res, post_res), (eval_domain, post_process))):
            func_res = func(domain, **kwargs)
            hparams = func_res.get('hparams', hparams)
            res[domain] = {'hparams': hparams, 'metrics': func_res['metrics']}
    return {'eval_res': eval_res, 'post_res': post_res, 'hparams': hparams}

def test_hparam(hparam, values, **kwargs):
    kwargs_dict = dict(kwargs)
    eval_res, post_res = [], []
    for v in tqdm(values):
        kwargs_dict[hparam] = v
        res = eval_all(**kwargs_dict)
        eval_res.append(res['eval_res'])
        post_res.append(res['post_res'])
    test_res = eval_res, post_res, hparam, values
    plot_all(*test_res)
    final_hparams = res['hparams']
    final_hparams.pop(hparam)
    print(final_hparams)

def plot_per_domain(res_dicts, hparam, values, title):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    fig.suptitle(title, fontsize=20)

    for i, domain in enumerate(['res', 'lap']):
        data = [d[domain]['metrics'] for d in res_dicts]
        df = pd.DataFrame(data, index=pd.Index(values, name=hparam))
        axs[i].set_yticks(np.linspace(.1, .9, num=33))
        axs[i].yaxis.set_tick_params(labelbottom=True)
        sns.lineplot(data=df, ax=axs[i]).set_title(domain_names[domain])

def plot_all(eval_res, post_res, hparam, values):
    data = []
    for res_dicts in eval_res, post_res:
        for i, domain in enumerate(['res', 'lap']):
            for res_dict, value in zip(res_dicts, values):
                for metric, score in res_dict[domain]['metrics'].items():
                    data.append({
                                hparam: value,
                                'domain': domain_names[domain],
                                'Metric': metric,
                                'score': score,
                                'Lemmatized': res_dicts == post_res})
                    
    fig = px.line(data, x=hparam, y='score', facet_col='domain', 
            line_dash='Lemmatized', color='Metric', line_shape='spline', hover_data={
                'Lemmatized': False,
                hparam: False,
                'domain': False,
                'Metric': True,
                'score': ":.3f"}).update_layout(title_text=f"Effect of '{hparam}' Value", title_x=0.5, hoverlabel=dict(
                    font_size=12,
                    font_family="Rockwell"),
                    font=dict(family="Courier New, monospace", size=18))\
            .update_traces(mode="markers+lines", hovertemplate="%{customdata[2]}=%{y:.3f}<extra></extra>")\
            .update_xaxes(showgrid=False, showspikes=True)\
            .show("notebook")

def apply_pattern(P1):
    def apply(text):
        delim = ' ' if text[-1] in ('.', '!', '?') else '. '
        return delim.join([text, P1])
    return apply

def mlm_splits(ds, pattern):
    P = apply_pattern(pattern)
    split_point = int(len(ds) * 0.7)
    mlm_train = ds[:split_point]
    mlm_test = ds[split_point:]
    with open('train.txt', 'w') as f:
        f.writelines(P(x) + '\n' for x, *_ in mlm_train)
    with open('test.txt', 'w') as f:
        f.writelines(P(x) + '\n' for x, *_ in mlm_test)

# run_mlm.py --seed=42 --num_train_epochs=2 --learning_rate=5e-04 --line_by_line \
#     --output_dir=. --train_file=train.txt --validation_file=test.txt --per_device_train_batch_size=1 \
#     --model_type=roberta --model_name_or_path=roberta-base --do_train --do_eval \
#     --overwrite_output_dir --overwrite_cache --evaluation_strategy=epoch