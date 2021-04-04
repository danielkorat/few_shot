from os import makedirs
import os
from numpy.lib.shape_base import hsplit
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
import seaborn as sns
from extract_aspects import extract_aspects_from_sentence, get_fm_pipeline

logging.disable(logging.WARNING)

from tqdm import tqdm as tq

def tqdm(iter, **kwargs):
    return tq(list(iter), kwargs, position=0, leave=True, file=stdout)

DOMAIN_NAMES = 'rest', 'lap'
DOMAIN_TO_STR = {'rest': 'Restaurants', 'lap': 'Laptops'}
DATA_DIR = 'https://raw.githubusercontent.com/IntelLabs/nlp-architect/libert/nlp_architect/models/libert/data/'
CSV_DATA_DIR = DATA_DIR + 'csv/spacy/domains_all/'
JSON_DATA_DIR = DATA_DIR + 'Dai2019/semeval14/'

# Mask Patterns
P1 = "So, the <mask> is the interesting aspect."
P2 = "So, the interesting aspect is <mask>."
P3 = "So, the <mask> are the interesting aspect."
P4 = "So, this is my opinion on <mask>."
P5 = "So, my review focuses on the <mask>."
P6 = "So, the <mask> is wonderful."
P7 = "So, the <mask> is awful."
P8 = "So, the main topic is the <mask>."
P9 = "So, it is all about the <mask>."
P10 = "So, I am talking about the <mask>."

PATTERNS = {'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7, 'P8': P8, 'P9': P9, 'P10': P10}

P_B1 = "So the <aspect> was <mask>."
P_B2 = "In summary, the <aspect> was <mask>."
P_B3 = "All in all, the <aspect> was <mask>."
P_B4 = "<mask>, the aspect is <aspect>."
P_B5 = "<mask>, the aspect in my review is <aspect>."
P_B6 = "<mask>, the topic of my review is <aspect>."
P_B7 = "Is it true that the aspect is <aspect>? <mask>."

PATTERNS_B = {'P_B1': P_B1, 'P_B2': P_B2, 'P_B3': P_B3, 'P_B4': P_B4, 'P_B5': P_B5, 'P_B6': P_B6, 'P_B7': P_B7}


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


def load_all_datasets(verbose=False, train_size=200):
    makedirs('data', exist_ok=True)
    if not os.path.exists(f"data/lap_train_{train_size}.json"):   
        for domain_name, domain in zip(DOMAIN_NAMES, ('restaurants', 'laptops')):
            ds = load_dataset(f"{domain}.csv", f"{domain}/{domain}_train_sents.json")
            if verbose:
                print(f'{domain} (size={len(ds)}):\n')
                for ex in ds[:5]:
                    print(ex[0], ex[3])

            train, test = ds[:train_size], ds[train_size:]
            print(f"{domain} size (train/test): {len(train)}/{len(test)}")

            print("Writing datest to json...")
            with open(f"data/{domain_name}_train_{train_size}.json", 'w') as train_f:
                json.dump(train, train_f, indent=2)
            with open(f"data/{domain_name}_test_{train_size}.json", 'w') as test_f:
                json.dump(test, test_f, indent=2)
    else:
        print("Loading dataset from json...")
    return {domain_name: {split: json.load(open(f"data/{domain_name}_{split}_{train_size}.json")) for split in ('train', 'test')} \
        for domain_name in DOMAIN_NAMES}


def run_ds_examples(ds, model_name, pattern_name, **kwargs):
    print(f"Pattern: {kwargs['pattern']}\n")
    for i, (text, tokens, gold_bio, aspects) in tqdm(enumerate(ds)):
        preds, valid_preds, pred_bio, _, _ = extract_aspects_from_sentence(PATTERNS=PATTERNS, text=text, tokens=tokens, model_name=model_name, pattern_name=pattern_name, **kwargs)
        print(i, text)
        print(tokens)
        print(f'gold: {aspects}\ngold_bio: {gold_bio}\nvalid_preds: {valid_preds}\npreds: {preds}\npred_bio: {pred_bio}\n')
    

def metrics(gold, preds, domain, verbose=False, **kwargs):
    F, P, R, conf = (f(gold, preds) for f in (f1_score, precision_score,\
                     recall_score, performance_measure))
    if verbose:
        print(f'{domain}')
        print(f'F1: {round(F, 3):.3f}, P: {P:.3f}, R: {R:.3f}, {conf}')

    return {'Precision': round(P,3), 'Recall': round(R,3), 'F1': round(F,3)}


def post_eval(ds_dict, domain, thresh=-1, **kwargs):
    with open(f'{domain}.pkl', 'rb') as f:
        _, all_mask_preds = pickle.load(f)

    nlp = spacy.load("en_core_web_sm", \
                     disable=["parser", "ner", "entity_linker", "textcat",
                              "entity_ruler", "sentencizer",
                              "merge_noun_chunks", "merge_entities",
                              "merge_subtokens"])
    
    # TODO: lemmatize sentences, not token lists

    all_preds_bio, all_gold_bio = [], []
    for (text, tokens, gold_bio, aspects), mask_preds in \
        zip(ds_dict[domain]['train'], all_mask_preds):

        pred_lems, valid_preds, token_lems = [], [], []
        for t in tokens:
            toks = list(nlp(t))
            token_lems.append(toks[0].lemma_ if toks else [''])

        valid_idx = set()
        for pred in mask_preds:
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


def run_example(text, tokens, top_k=10, thresh=-1, target=True, **kwargs):
    hparams = locals()
    for v in 'text', 'tokens', 'kwargs':
        hparams.pop(v)
    hparams.update(kwargs)

    delim = ' ' if text[-1] in ('.', '!', '?') else '. '
    pattern = PATTERNS[kwargs['pattern_names'][0]]
    fm_pipeline = get_fm_pipeline(kwargs['model_name'])
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


def eval_ds_run_example(ds_dict, domain, model_name, pattern_names, exper_str, test_limit=None, **kwargs):
    all_preds_bio, all_preds, all_valid_preds, all_mask_preds, all_gold_bio = [], [], [], [], []
    for text, tokens, gold_bio, aspects in tqdm(ds_dict[domain]['test'][:test_limit]):
        preds, valid_preds, pred_bio, mask_preds, hparams = \
            run_example(text=text, tokens=tokens, model_name=model_name, pattern_names=pattern_names,  **kwargs)                                                  
        all_preds.append(preds)
        all_valid_preds.append(valid_preds)
        all_preds_bio.append(pred_bio)
        all_mask_preds.append(mask_preds)
        all_gold_bio.append(gold_bio)

    makedirs('predictions', exist_ok=True)
    with open(f'predictions/{domain}_{exper_str}.json', 'w') as f:
        json.dump((all_preds, all_mask_preds), f, indent=2)

    return {'metrics': metrics(all_gold_bio, all_preds_bio, domain, **kwargs), 'hparams': hparams}


def eval_ds(ds_dict, domain, pattern_names, pattern_names_B, exper_str, test_limit=None, **kwargs):
    all_preds_bio, all_preds, all_valid_preds, all_gold_bio = [], [], [], []
    for text, tokens, gold_bio, aspects in tqdm(ds_dict[domain]['test'][:test_limit]):
        preds, pred_bio, hparams = extract_aspects_from_sentence(PATTERNS=PATTERNS, PATTERNS_B=PATTERNS_B, 
                                                        text=text, tokens=tokens, model_name=kwargs['model_name'],
                                                        pattern_names=pattern_names, pattern_names_B=pattern_names_B, **kwargs)                                                  
        all_preds.append(preds)
        all_preds_bio.append(pred_bio)
        all_gold_bio.append(gold_bio)

    # write predictions to file
    makedirs('predictions', exist_ok=True)
    with open(f'predictions/{domain}_{exper_str}.json', 'w') as f:
        json.dump((all_preds), f, indent=2)

    return {'metrics': metrics(all_gold_bio, all_preds_bio, domain, **kwargs), 'hparams': hparams}

    # makedirs('predictions', exist_ok=True)
    # with open(f'predictions/{domain}.pkl', 'wb') as f:
    #     pickle.dump((all_preds, all_mask_preds), f)

    # return {'metrics': metrics(all_gold_bio, all_preds_bio, domain, **kwargs), 'hparams': hparams}


def evaluate(exper_name='', post=False, **kwargs):
    if exper_name:
        exper_name = '_' + exper_name
    ds_dict = load_all_datasets()
    all_res = {}
    exper_str = f"{kwargs['model_name'].replace('models/', '')}{exper_name}"
    makedirs('eval', exist_ok=True)
    with open(f"eval/{exper_str}.txt", 'w') as eval_f:
        for i, domain in enumerate(['rest', 'lap']):
            res = eval_ds_run_example(ds_dict=ds_dict, domain=domain, exper_str=exper_str, **kwargs)
            all_res[domain] = res
            p, r, f1 = [f"{100. * res['metrics'][m]:.2f}" for m in ('Precision', 'Recall', 'F1')]

            print(f'Test Domain: {domain}', file=eval_f)
            writer = csv.writer(eval_f, delimiter="\t")
            writer.writerows([['Metric', 'Score'],
                                ['P', p],
                                ['R', r],
                                ['F1', f1]])
            print("________________\n", file=eval_f)

            if i == 1:
                print(res['hparams'], file=eval_f)

            if post:
                # reads output files generated by eval_ds()
                post_metrics = post_eval(ds_dict, domain, **kwargs)
                print(f"Post-Evaluation results on '{domain}' train data:\n{post_metrics}\n")
    return all_res


def apply_pattern(P1):
    def apply(text):
        delim = ' ' if text[-1] in ('.', '!', '?') else '. '
        return delim.join([text, P1])
    return apply

def apply_pattern_and_replace_mask(f, P, x, unique_count, count, replace_with):
    unique_count += 1
    for replacement in replace_with:
        count += 1
        line = P(x).replace('<mask>', replacement) + '\n'
        f.write(line)
    return unique_count, count

def create_mlm_train_sets(ds_dict, labelled_amount, sample_selection, **kwargs):
    pattern_name = kwargs['pattern_names'][0]   
    P = apply_pattern(PATTERNS[pattern_name])
    makedirs('mlm_data', exist_ok=True)
    actual_labelled_amounts = {}

    for domain in 'rest', 'lap':
        count, unique_count = 0, 0
        train_samples = ds_dict[domain]['train']
        train_out_path = f'mlm_data/{domain}_train_{labelled_amount}_{pattern_name}.txt'

        with open(train_out_path, 'w') as f:
            # Use only samples with aspects
            if sample_selection == 'take_positives':
                for x, *_, aspects in train_samples[:labelled_amount]:
                    if aspects:
                        unique_count, count = \
                            apply_pattern_and_replace_mask(f, P, x, unique_count, count, replace_with=aspects)

            # Use only samples with aspects, match required labelled amount
            elif sample_selection == 'match_positives':
                for x, *_, aspects in train_samples:
                    if aspects:
                        unique_count, count = \
                            apply_pattern_and_replace_mask(f, P, x, unique_count, count, replace_with=aspects)
                    if unique_count == labelled_amount:
                        break
                assert unique_count == labelled_amount

            # Insert 'NONE' as an aspect when there are no aspects
            elif sample_selection == 'negatives_with_none':
                for x, *_, aspects in train_samples:
                    if aspects:
                        unique_count, count = \
                            apply_pattern_and_replace_mask(f, P, x, unique_count, count, aspects, replace_with=aspects)
                    else:
                        unique_count, count = \
                            apply_pattern_and_replace_mask(f, P, x, unique_count, count, replace_with=['NONE'])

        actual_labelled_amounts[domain] = (unique_count, count)
    return actual_labelled_amounts


def plot_per_domain(res_dicts, hparam, values, title):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    fig.suptitle(title, fontsize=20)

    for i, domain in enumerate(['rest', 'lap']):
        data = [d[domain]['metrics'] for d in res_dicts]
        df = pd.DataFrame(data, index=pd.Index(values, name=hparam))
        axs[i].set_yticks(np.linspace(.1, .9, num=33))
        axs[i].yaxis.set_tick_params(labelbottom=True)
        sns.lineplot(data=df, ax=axs[i]).set_title(DOMAIN_NAMES[domain])


def plot_few_shot(train_domain, plot_data, train_hparams, actual_num_labelled=None):
    data = []

    # Format Hyperparameters
    hp_dict = plot_data[0]['lap']['hparams']
    hp_dict.update(train_hparams)
    hp = list(hp_dict.items())
    hparams = '<br>'.join([(', '.join([f"{k}: {v}" for k, v in hp[i: i + 9]])) for i in range(0, len(hp), 9)])
    if actual_num_labelled:
        hparams += ', actual_num_labelled: ' + str(actual_num_labelled)

    for test_domain in 'lap', 'rest':
        for num_labelled, res_dict in plot_data.items():
            for metric, score in res_dict[test_domain]['metrics'].items():
                data.append({
                    'num_labelled': num_labelled,
                    'test_domain': DOMAIN_TO_STR[test_domain],
                    'metric': metric,
                    'score': score
                    })

    px.line(data, x='num_labelled', y='score', facet_col='test_domain', color='metric', line_shape='spline',
            hover_data={
                'num_labelled': False,
                'test_domain': False,
                'metric': True,
                'score': ":.3f"})\
                .update_layout(title_text=f"Effect of num_labelled, trained on {DOMAIN_TO_STR[train_domain]}<br>" +\
                    '<span style="font-size: 12px;">' + f"{hparams} </span>",
                    margin=dict(t=220),
                    title_x=0.5,
                    font=dict(family="Courier New, monospace", size=18),
                    hoverlabel=dict(
                        font_size=12,
                        font_family="Rockwell")
                        )\
                .update_traces(mode="markers+lines", hovertemplate="%{customdata[1]}=%{y:.3f}<extra></extra>")\
                .update_xaxes(showgrid=False, showspikes=True)\
                .show()
