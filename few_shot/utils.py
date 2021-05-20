from patterns import ROOT
from os import makedirs, write
import os
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
from extract_aspects import extract_aspects, generate_bio, get_fm_pipeline
import spacy
from patterns import PATTERNS, SCORING_PATTERNS

logging.disable(logging.WARNING)

from tqdm import tqdm as tq

print("LOADING SPACY MODEL ----------------------")
spacy_model = spacy.load('en_core_web_sm')

def tqdm(iter, **kwargs):
    return tq(list(iter), kwargs, position=0, leave=True, file=stdout)

DOMAIN_NAMES = 'rest', 'lap'
DOMAIN_TO_STR = {'rest': 'Restaurants', 'lap': 'Laptops'}
DATA_DIR = 'https://raw.githubusercontent.com/IntelLabs/nlp-architect/libert/nlp_architect/models/libert/data/'
CSV_DATA_DIR = DATA_DIR + 'csv/spacy/domains_all/'
JSON_DATA_DIR = DATA_DIR + 'Dai2019/semeval14/'


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
    makedirs(str(ROOT / 'data'), exist_ok=True)
    if not os.path.exists(ROOT / "data" / f"lap_train_{train_size}.json"):   
        for domain_name, domain in zip(DOMAIN_NAMES, ('restaurants', 'laptops')):
            ds = load_dataset(f"{domain}.csv", f"{domain}/{domain}_train_sents.json")
            if verbose:
                print(f'{domain} (size={len(ds)}):\n')
                for ex in ds[:5]:
                    print(ex[0], ex[3])

            train, test = ds[:train_size], ds[train_size:]
            print(f"{domain} size (train/test): {len(train)}/{len(test)}")

            print("Writing datest to json...")
            with open(ROOT / "data" / "{domain_name}_train_{train_size}.json", 'w') as train_f:
                json.dump(train, train_f, indent=2)
            with open(ROOT / "data" / f"{domain_name}_test_{train_size}.json", 'w') as test_f:
                json.dump(test, test_f, indent=2)
    else:
        print("Loading dataset from json...")
    return {domain: {split: json.load(open(ROOT / "data" / f"{domain}_{split}_{train_size}.json")) \
        for split in ('train', 'test')} for domain in DOMAIN_NAMES}


def run_ds_examples(ds, model_name, pattern_name, **kwargs):
    print(f"Pattern: {kwargs['pattern']}\n")
    for i, (text, tokens, gold_bio, aspects) in tqdm(enumerate(ds)):
        preds, valid_preds, pred_bio, _, _ = extract_aspects(text=text, tokens=tokens, pattern_name=pattern_name, **kwargs)
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


def eval_ds(ds_dict, test_domain, pattern_names, model_names, scoring_model_names=None,
        scoring_patterns=None, test_limit=None, **kwargs):

    test_data = ds_dict[test_domain]['test'][:test_limit]
    all_gold_bio = []

    # in case of pre-trained model evaluation
    if len(model_names) != len(pattern_names):
        model_names *= len(pattern_names)

    scoring_pipeline = get_fm_pipeline(scoring_model_names[0]) if scoring_model_names else None
  
    all_preds_list = []
    for i, (model_name, pattern_name) in enumerate(zip(model_names, pattern_names)):

        print(f"Evaluating ({model_name}, {pattern_name})" + f"with scoring model {scoring_model_names[0]}" if scoring_model_names else "")
        fm_pipeline = get_fm_pipeline(model_name)

        all_preds_bio, all_preds = [], []

        for text, tokens, gold_bio, aspects in tqdm(test_data):

            preds, preds_bio, hparams = extract_aspects(fm_pipeline=fm_pipeline, scoring_pipeline=scoring_pipeline,
                text=text, tokens=tokens, pattern_names=(pattern_name,), scoring_patterns=scoring_patterns, **kwargs)

            all_preds.append(preds)
            all_preds_bio.append(preds_bio)

            if i == 0:
                all_gold_bio.append(gold_bio)

        # write predictions to file
        makedirs(ROOT / 'predictions', exist_ok=True)
        with open(ROOT / 'predictions' / f'{model_name}_{pattern_name}_{test_domain}.json', 'w') as f:
            json.dump((all_preds), f, indent=2)

        all_preds_list.append(all_preds)

    if len(pattern_names) == 1:
        final_preds_bio = all_preds_bio
    
    # Merge predictions in case of multiple patterns
    else:
        final_preds_bio = []
        for (_, tokens, *_), all_preds in zip(test_data, zip(*all_preds_list)):
            bio = generate_bio(tokens=tokens, preds=list({p for preds in all_preds for p in preds}))
            final_preds_bio.append(bio)

    preds_fname = f"{model_names[0]}_{test_domain}"
    if len(model_names) > 1:
        preds_fname = preds_fname.replace(pattern_names[0], '+'.join(pattern_names))

    with open(f"predictions/{preds_fname}.json", 'w') as f:
        json.dump((final_preds_bio), f, indent=2)

    return {'metrics': metrics(all_gold_bio, final_preds_bio, test_domain, **kwargs), 'hparams': hparams}


def evaluate(test_domains, pattern_names, model_names, **kwargs):
    ds_dict = load_all_datasets()
    all_res = {}
    
    eval_fname = model_names[0]

    if len(model_names) > 1:
        eval_fname = eval_fname.replace(pattern_names[0], '+'.join(pattern_names))

    makedirs(ROOT / 'eval', exist_ok=True)
    with open(ROOT / "eval" / f"{eval_fname}.txt", 'w') as eval_f:
        for i, test_domain in enumerate(test_domains):
            res = eval_ds(ds_dict=ds_dict, model_names=model_names, test_domain=test_domain, 
                pattern_names=pattern_names, **kwargs)
            all_res[test_domain] = res
            p, r, f1 = [f"{100. * res['metrics'][m]:.2f}" for m in ('Precision', 'Recall', 'F1')]

            print(f'Test Domain: {test_domain}', file=eval_f)
            writer = csv.writer(eval_f, delimiter="\t")
            writer.writerows([['Metric', 'Score'],
                                ['P', p],
                                ['R', r],
                                ['F1', f1]])
            print("________________\n", file=eval_f)

        print(res['hparams'], file=eval_f)
    return all_res


def apply_pattern(P1):
    def apply(text):
        delim = ' ' if text[-1] in ('.', '!', '?') else '. '
        return delim.join([text, P1])
    return apply

def replace_mask(train_samples, path, P, none_replacement=None, limit=None, require_aspects=True):
    count, unique_count = 0, 0

    with open(path, 'w') as f:
        for x, *_, aspects in train_samples[:limit]:
            P_x = P(x)

            if aspects or not require_aspects:
                unique_count += 1

                replacements = aspects if aspects else [none_replacement]

                for replacement in replacements:
                    count += 1
                    line = P_x.replace('<mask>', replacement) + '\n'
                    f.write(line)

    return {'unique': unique_count, 'total': count}

def replace_mask_scoring_pattern(f, P, x, replace_aspects, replace_mask_token):
    count, unique_count = 0, 0
    unique_count += 1
    P_x = P(x)
    
    for replace_asp in replace_aspects:
        count += 1
        pattern = P_x.replace('<aspect>', replace_asp)
        line = pattern.replace('<mask>', replace_mask_token) + '\n'
        f.write(line)
    return unique_count, count    

def create_mlm_train_sets(datasets, num_labelled, train_domains, masking_strategy, **kwargs):
    actual_num_labelled = {}
    makedirs(ROOT / 'mlm_data', exist_ok=True)

    pattern = "Is there sentiment towards <aspect> in the previous sentence? <mask>"

    for train_domain in train_domains:

        train_samples = datasets[train_domain]['train']
        exper_str = f"{train_domain}_{num_labelled}_{masking_strategy}"
        out_path = str(ROOT / 'mlm_data' / f'{exper_str}')

        if masking_strategy == 'aspect_scoring':
            count, unique_count = 0, 0

            with open(out_path + '.csv', 'w') as f:
                with open(out_path + '.txt', 'w') as txt_f:

                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(["word", "is_aspect", "text"])

                    counts = 0
                    for text, *_, gold_aspects in train_samples[:num_labelled]:                    
                        # add positive label examples
                        label_pairs = [[gold_asp, "1"] for gold_asp in gold_aspects]

                        # add negative label examples: extract non-aspect nouns     
                        nouns = {ent.text for ent in spacy_model(text) if ent.pos_ == 'NOUN'}
                        for non_asp in nouns - set(gold_aspects):
                            label_pairs.append([non_asp, "0"])                            

                        # write labelled examples to file
                        if label_pairs: # skips examples without gold aspects and nouns
                            writer.writerows([[*label_pair, text] for label_pair in label_pairs])

                            for label_pair in label_pairs: 
                                txt_f.write(text + ' ' + pattern.replace("<aspect>", label_pair[0]) + '\n')

                            unique_count += 1
                            count += len(label_pairs)

                    counts = {'unique': unique_count, 'total': count}

        actual_num_labelled[train_domain] = counts                                 

    return actual_num_labelled


def plot_per_domain(res_dicts, hparam, values, title):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    fig.suptitle(title, fontsize=20)

    for i, domain in enumerate(['rest', 'lap']):
        data = [d[domain]['metrics'] for d in res_dicts]
        df = pd.DataFrame(data, index=pd.Index(values, name=hparam))
        axs[i].set_yticks(np.linspace(.1, .9, num=33))
        axs[i].yaxis.set_tick_params(labelbottom=True)
        sns.lineplot(data=df, ax=axs[i]).set_title(DOMAIN_NAMES[domain])


def plot_few_shot(train_domain, test_domains, plot_data, train_hparams={}, actual_num_labelled=None, **kwargs):
    data = []

    # Format Hyperparameters
    hp_dict = plot_data[0][train_domain]['hparams']
    hp_dict.update(train_hparams)
    hp = list(hp_dict.items())
    hparams = '<br>'.join([(', '.join([f"{k}: {v}" for k, v in hp[i: i + 9]])) for i in range(0, len(hp), 9)])
    if actual_num_labelled:
        hparams += ', actual_num_labelled: ' + str(actual_num_labelled)

    for test_domain in test_domains:
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
