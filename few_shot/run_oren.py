from patterns import ROOT
from utils import load_all_datasets, evaluate, create_mlm_train_sets, plot_few_shot, PATTERNS, SCORING_PATTERNS, eval_ds, create_asp_only_data_files
from run_pattern_mlm import main as run_pattern_mlm
import os


def pattern_mlm_preprocess(labelled_amounts, **kwargs):
    datasets = load_all_datasets()
    res = {}
    # Prepare Pattern-MLM Training
    # Write splits to '/mlm_data'
    for num_labelled in labelled_amounts:
        amounts = create_mlm_train_sets(datasets, num_labelled, **kwargs)
        res[num_labelled] = amounts
    return res
     
     
def train_mlm(train_domain, num_labelled, pattern_name, seed=42, lr=1e-05, max_seq=256, max_steps=1000, batch_size=16,
            validation=None, model_type='roberta', model_name='roberta-base', **kwargs):
    hparams = locals()
    for v in 'train_domain', 'num_labelled', 'kwargs':
        hparams.pop(v)

    os.makedirs('models', exist_ok=True)
    if pattern_name.startswith('P_B'):
        output_dir = f"models/scoring_{pattern_name}_{train_domain}_{num_labelled}"
    else:
        output_dir = f"models/p-mlm_model_{train_domain}_{num_labelled}"
    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'

    # if pattern_name.startswith('P_B'):
    #     pattern = SCORING_PATTERNS[pattern_name]
    # else:
    #     pattern = PATTERNS[pattern_name]
    run_pattern_mlm([
    "--pattern", pattern_name,
     "--model_cls", "RobertaForMLMWithCE",
    "--seed", str(seed),
    # "--num_train_epochs", "1",
    "--learning_rate", str(lr),
    "--max_seq_length", str(max_seq),
    "--max_steps", str(max_steps),
    "--train_file", "mlm_data/" + train_domain + f'_train_{num_labelled}_{pattern_name}.txt',
    "--per_device_train_batch_size", str(batch_size),
    # "--validation_file", "mlm_data/" + validation,
    # "--do_eval", "--validation_file", "mlm_data/rest_test.txt",
    # "--evaluation_strategy", "epoch",
    "--line_by_line", "--output_dir", output_dir,
    "--model_type", model_type, "--model_name_or_path", model_name,
    "--do_train", "--overwrite_output_dir", "--overwrite_cache"])
    return output_dir, hparams


def train_scoring_pattern(labelled_amounts, **kwargs):

    actual_num_labelled = pattern_mlm_preprocess(labelled_amounts,  **kwargs)
    print("actual_num_labelled: ", actual_num_labelled)
    trained_models = []
    for train_domain in kwargs['train_domains']:
        for num_labelled in labelled_amounts:
            print(f"\n{'-' * 50}\n\t\t  Num. Labelled: {num_labelled}\n{'-' * 50}")
            trained_model, hparams = train_mlm(train_domain, num_labelled, pattern_name=kwargs['pattern_names'][0], **kwargs)
            trained_models.append(trained_model)
            print("Finished model training")
    return trained_models


def eval(scoring_models):
    pattern_kwargs = dict(pattern='P5', top_k=10, step_1_nouns_only=True)
    data = load_all_datasets(train_size=2200)
    model_names=['roberta-base']
   
    #pattern_groups=(['P1','P2'], ['P1','P2','P3'], ['P1','P2','P3','P4'],['P1','P2','P3','P4','P5'],['P1','P2','P3','P4','P5','P6'],
    # ['P1','P2','P3','P4','P5','P6','P7'],['P1','P2','P3','P4','P5','P6','P7','P8'])
    #pattern_groups=(['P1','P2'],['P1','P2','P3'])
    pattern_groups=(['P1'],)
    scoring_patterns=(['P_B13'])
    test_domain='rest'
    
    #scoring_patterns=None
    for pattern_names in pattern_groups:
        eval_results = eval_ds(data, test_domain=test_domain, pattern_names=pattern_names, model_names=model_names, scoring_model_names=scoring_models, 
            scoring_patterns=scoring_patterns, **pattern_kwargs)
        print("Stats:", eval_results['metrics'], "   Patterns: ", pattern_names)


def main():
   
    scoring_models = train_scoring_pattern(pattern_names=('P_B13',), labelled_amounts=range(64, 65), sample_selection='negatives_with_none',
       model_name='roberta-base', train_domains=['rest'], test_domains=['lap'],
       masking_strategy='aspect_scoring', max_steps=5, test_limit=5)

    eval(scoring_models)

#    create_asp_only_data_files("rest/asp+op/res_all.json", "rest/asp/res_all.json") 

    # scoring_models = ['scoring_P_B13_rest_100']
    #scoring_models = ['scoring_P_B15_rest_16']
    #scoring_models = ['scoring_P_B12_lap_100']
    #scoring_models = ['scoring_P_B14_lap_100']
    #scoring_models = (['scoring_P_B12_rest_16', 'scoring_P_B13_rest_16','scoring_P_B14_rest_16','scoring_P_B15_rest_16'])
    #scoring_models = (['scoring_P_B12_rest_100', 'scoring_P_B13_rest_100', 'scoring_P_B14_rest_100'])
    #scoring_models = (['scoring_P_B12_rest_100'])


if __name__ == "__main__":
    main()
    #plot_few_shot('lap', *pickle.load(open(f'lap_plot_data.pkl', 'rb')), "dfgdf")
