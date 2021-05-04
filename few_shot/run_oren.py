from patterns import ROOT
from utils import load_all_datasets, evaluate, create_mlm_train_sets, plot_few_shot, PATTERNS, SCORING_PATTERNS, eval_ds
from run_pattern_mlm import main as run_pattern_mlm
from run import train_mlm

def pattern_mlm_preprocess(labelled_amounts, **kwargs):
    datasets = load_all_datasets()
    res = {}
    # Prepare Pattern-MLM Training
    # Write splits to '/mlm_data'
    for num_labelled in labelled_amounts:
        amounts = create_mlm_train_sets(datasets, num_labelled, **kwargs)
        res[num_labelled] = amounts
    return res

def train_scoring_pattern(labelled_amounts, **kwargs):
    actual_num_labelled = pattern_mlm_preprocess(labelled_amounts, **kwargs)
    print("actual_num_labelled: ",actual_num_labelled) 
    for train_domain in kwargs['train_domains']:
        for num_labelled in labelled_amounts:
            print(f"\n{'-' * 50}\n\t\t  Num. Labelled: {num_labelled}\n{'-' * 50}")
            p_mlm_model, hparams = train_mlm(train_domain, num_labelled, pattern_name=kwargs['pattern_names'][0], **kwargs)    
            print("Finished model training")

def eval_pretrained():
    pattern_kwargs = dict(pattern='P5', top_k=10, step_1_nouns_only=True)
    data = load_all_datasets(train_size=200)
    model_names=['roberta-base']
    #scoring_model_names = ['p-mlm_model_scoring_P_B12_rest_100']
    #scoring_model_names = ['p-mlm_model_scoring_P_B12_lap_100']
    scoring_model_names = None
    #pattern_groups=(['P1','P2'], ['P1','P2','P3'], ['P1','P2','P3','P4'],['P1','P2','P3','P4','P5'],['P1','P2','P3','P4','P5','P6'],['P1','P2','P3','P4','P5','P6','P7'],['P1','P2','P3','P4','P5','P6','P7','P8'])
    #pattern_groups=(['P1','P2'],['P1','P2','P3'])
    pattern_groups=(['P1'],)
    #scoring_patterns=(['P_B12'])
    scoring_patterns=None
    for pattern_names in pattern_groups:
        eval_results = eval_ds(data, test_domain='lap', pattern_names=pattern_names, model_names=model_names,scoring_model_names=scoring_model_names, 
            scoring_patterns=scoring_patterns, **pattern_kwargs)
        print("Stats:", eval_results['metrics'], "   Patterns: ", pattern_names)


def main():
    
    # eval_pretrained()
   
    train_scoring_pattern(pattern_names=('P_B12',), labelled_amounts=range(100, 101), sample_selection='negatives_with_none',
        model_name='roberta-base', train_domains=['rest'], test_domains=['rest'],
        masking_strategy='aspect_scoring', max_steps=5, test_limit=5)    

if __name__ == "__main__":
    main()



