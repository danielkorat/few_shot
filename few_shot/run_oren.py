from patterns import ROOT
from utils import load_all_datasets, evaluate, create_mlm_train_sets, plot_few_shot, PATTERNS, SCORING_PATTERNS, eval_ds
from run_pattern_mlm import main as run_pattern_mlm
import os
import pickle

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
        output_dir = f"models/p-mlm_model_scoring_{train_domain}_{num_labelled}"
    else:
        output_dir = f"models/p-mlm_model_{train_domain}_{num_labelled}"
    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'

    pattern = SCORING_PATTERNS[pattern_name] if pattern_name.startswith('P_B') else PATTERNS[pattern_name]

    # if pattern_name.startswith('P_B'):
    #     pattern = SCORING_PATTERNS[pattern_name]
    # else:
    #     pattern = PATTERNS[pattern_name]
    run_pattern_mlm([
    "--pattern", pattern,
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
    actual_num_labelled = pattern_mlm_preprocess(labelled_amounts, **kwargs)
    print("actual_num_labelled: ",actual_num_labelled) 
    for train_domain in kwargs['train_domains']:
        for num_labelled in labelled_amounts:
            print(f"\n{'-' * 50}\n\t\t  Num. Labelled: {num_labelled}\n{'-' * 50}")
            p_mlm_model, hparams = train_mlm(train_domain, num_labelled, pattern_name=kwargs['pattern_names'][0], **kwargs)    
            print("Finished model training")

def eval_pretrained():
    pattern_kwargs = dict(pattern='P5', top_k=10)
    data = load_all_datasets(train_size=200)
    model_names=['roberta-base']
    scoring_model_names = ['p-mlm_model_scoring_rest_100']
    pattern_groups=(['P1','P2'], ['P1','P2','P3'], ['P1','P2','P3','P4'],['P1','P2','P3','P4','P5'],['P1','P2','P3','P4','P5','P6'],['P1','P2','P3','P4','P5','P6','P7'],['P1','P2','P3','P4','P5','P6','P7','P8'])
    #pattern_groups=(['P1','P2'],['P1','P2','P3'])
    #pattern_groups=(['P1'],)
    scoring_patterns=(['P_B12'])
    for pattern_names in pattern_groups:
        eval_results = eval_ds(data, test_domain='rest', pattern_names=pattern_names, model_names=model_names,scoring_model_names=scoring_model_names, 
            scoring_patterns=scoring_patterns, **pattern_kwargs)
        print("Stats:", eval_results['metrics'], "   Patterns: ", pattern_names)


def main():
    
    eval_pretrained()
   

    # train_scoring_pattern(pattern_names=('P_B12',), labelled_amounts=range(100, 101), sample_selection='negatives_with_none',
    #     model_name='roberta-base', train_domains=['rest'], test_domains=['rest'],
    #     masking_strategy='aspect_scoring')#, max_steps=5, test_limit=5)    

if __name__ == "__main__":
    main()
    # plot_few_shot('lap', *pickle.load(open(f'lap_plot_data.pkl', 'rb')), "dfgdf")



