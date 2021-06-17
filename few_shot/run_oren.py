from numpy.lib.shape_base import split
from patterns import ROOT
from utils import load_datasets, create_mlm_train_sets, plot_few_shot, PATTERNS, SCORING_PATTERNS, eval_ds, create_asp_only_data_files, create_new_data_files
from run_pattern_mlm import main as run_pattern_mlm
import os
from itertools import product
from datetime import datetime

def pattern_mlm_preprocess(train_size, split, **kwargs):
    datasets =load_datasets(split, domain='rest',  train_size=100)
    res = {}
    # Prepare Pattern-MLM Training
    # Write splits to '/mlm_data'
    for num_labelled in train_size:        
        amounts = create_mlm_train_sets(datasets, split, num_labelled, **kwargs)
        res[num_labelled] = amounts
    return res
     
     
def train_mlm(split, train_domain, num_labelled, pattern_name, alpha, seed=42, lr=1e-05, max_seq=256, max_steps=1000, batch_size=16,
            validation=None, model_type='roberta', model_name='roberta-base', **kwargs):
    hparams = locals()
    for v in 'train_domain', 'num_labelled', 'kwargs':
        hparams.pop(v)

    os.makedirs('models', exist_ok=True)
    if pattern_name.startswith('P_B'):
        output_dir = f"models/scoring_{pattern_name}_{train_domain}_{num_labelled}_split{split}"
    else:
        output_dir = f"models/p-mlm_model_{train_domain}_{num_labelled}"
    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'

    # if pattern_name.startswith('P_B'):
    #     pattern = SCORING_PATTERNS[pattern_name]
    # else:
    #     pattern = PATTERNS[pattern_name]
    run_pattern_mlm([
    "--pattern", pattern_name,
    "--model_cls", "RobertaForMLMWithCE",
    "--alpha", str(alpha),
    "--seed", str(seed),
    # "--num_train_epochs", "1",
    "--learning_rate", str(lr),
    "--max_seq_length", str(max_seq),
    "--max_steps", str(max_steps),
    "--train_file", "mlm_data/" + train_domain + f'_train_{num_labelled}_{pattern_name}_split{split}.txt',
    "--per_device_train_batch_size", str(batch_size),
    # "--validation_file", "mlm_data/" + validation,
    # "--do_eval", "--validation_file", "mlm_data/rest_test.txt",
    # "--evaluation_strategy", "epoch",
    "--line_by_line", "--output_dir", output_dir,
    "--model_type", model_type, "--model_name_or_path", model_name,
    "--do_train", "--overwrite_output_dir", "--overwrite_cache"])
    return output_dir, hparams


def train_scoring_pattern(split, train_sizes, **kwargs):

    actual_num_labelled = pattern_mlm_preprocess(train_sizes, split,  **kwargs)
    print("actual_num_labelled: ", actual_num_labelled)
    trained_models = []
    for train_domain in kwargs['train_domains']:
        for num_labelled in train_sizes:
            print(f"\n{'-' * 55}\n\t\t  Num. Labelled: {num_labelled}\n{'-' * 55}")
            pattern_name = kwargs['pattern_names'][0]
            print(f"Training model: {pattern_name}_{train_domain}_{num_labelled}_split{split}")
            trained_model, hparams = train_mlm(split, train_domain, num_labelled, pattern_name=kwargs['pattern_names'][0], **kwargs)
            trained_models.append(trained_model)
            print("Finished model training")
    return trained_models


def eval(scoring_models, test_domain, split, train_size, scoring_pattern, pattern):
    pattern_kwargs = dict(pattern='P5', top_k=10, step_1_nouns_only=True)
    
    data = load_datasets(split, test_domain,  train_size) 
    model_names=['roberta-base']
   
    #pattern_groups=(['P1','P2'], ['P1','P2','P3'], ['P1','P2','P3','P4'],['P1','P2','P3','P4','P5'],['P1','P2','P3','P4','P5','P6'],
    # ['P1','P2','P3','P4','P5','P6','P7'],['P1','P2','P3','P4','P5','P6','P7','P8'])
    #pattern_groups=(['P1','P2'],['P1','P2','P3'])
    pattern_groups=([pattern],)
    scoring_patterns=([scoring_pattern])
    
    #scoring_patterns=None
    for pattern_names in pattern_groups:
        eval_results = eval_ds(split, data, test_domain=test_domain, pattern_names=pattern_names, model_names=model_names, scoring_model_names=scoring_models, 
            scoring_patterns=scoring_patterns, **pattern_kwargs)
        print("Stats:", eval_results['metrics'], "   Patterns: ", pattern_names)
    return eval_results['metrics']

def main(smoke):

    if smoke:
        alphas, domains, train_sizes = [0], ['rest', 'lap'], [16]
        kwargs = dict(max_steps=5, test_limit=5)
    else:
        # alphas = [0, .2, .4, .6, .8, 1]
        # domains = ['rest', 'lap']
        # train_sizes = [16, 32, 64, 100]
        
        alphas = [.6, ]
        domains = ['rest']
        splits = [1]
        train_sizes = [100]
        kwargs = {}
        #kwargs = dict(max_steps=5)

    pattern='P5'
    scoring_pattern = 'P_B13'
    timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M")

    for domain, alpha, train_size, split in product(domains, alphas, train_sizes, splits):
        scoring_models = train_scoring_pattern(split, pattern_names=(scoring_pattern,), 
            train_sizes=[train_size], sample_selection='negatives_with_none',
            model_name='roberta-base', train_domains=[domain], test_domains=[domain],
            masking_strategy='aspect_scoring', alpha=alpha, **kwargs)

        print("scoring_models:", scoring_models)

        res = eval(scoring_models, domain, split, train_size, scoring_pattern, pattern)

        with open(f'results_{timestamp}.txt', 'a') as f:
            f.write(f"domain: {domain}, scoring_pattern: {scoring_pattern}, alpha: {alpha}, train_size: {train_size}, split: {split}\n{res}\n\n")  

           
    #create_asp_only_data_files("rest/asp+op/rest_all.json", "rest/asp/rest_all.json")
    
    #create_new_data_files("data/rest/asp+op/rest_all.txt", "data/rest/asp+op/rest_all_sentences.txt", "data/rest/asp+op/rest_all.json")
    


def run_eval_only():

    domain = 'rest'
    splits = [1]
    train_size = '100'
    pattern='P5'
    scoring_pattern = 'P_B13'
    scoring_models = ["models/scoring_P_B13_rest_100_split1"] 
    timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M")

    for split in splits:
        res = eval(scoring_models, domain, split, train_size, scoring_pattern, pattern)
        with open(f'results_{timestamp}.txt', 'a') as f:
            f.write(f"domain: {domain}, scoring_pattern: {scoring_pattern}, split: {split}\n{res}\n\n")  

if __name__ == "__main__":
    #run_eval_only()
    main(smoke=False)
    #plot_few_shot('lap', *pickle.load(open(f'lap_plot_data.pkl', 'rb')), "dfgdf")
