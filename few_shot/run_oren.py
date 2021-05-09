from patterns import ROOT
from utils import load_all_datasets, create_mlm_train_sets, PATTERNS, SCORING_PATTERNS, eval_ds
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
    print("actual_num_labelled: ", actual_num_labelled) 
    for train_domain in kwargs['train_domains']:
        for num_labelled in labelled_amounts:
            print(f"\n{'-' * 50}\n\t\t  Num. Labelled: {num_labelled}\n{'-' * 50}")
            p_mlm_model, hparams = train_mlm(train_domain, num_labelled, pattern_name=kwargs['pattern_names'][0], **kwargs)    
            print("Finished model training")
    return p_mlm_model

def eval_model(model_names=['roberta-base'], scoring_model_names=None):
    pattern_kwargs = dict(pattern='P5', top_k=10, step_1_nouns_only=True)
    data = load_all_datasets(train_size=200)
    pattern_groups=(['P1'],)
    scoring_patterns=['P_B12']
    for pattern_names in pattern_groups:
        eval_results = eval_ds(data, test_domain='lap', pattern_names=pattern_names, model_names=model_names, 
            scoring_model_names=scoring_model_names, scoring_patterns=scoring_patterns, **pattern_kwargs)
        print("Stats:", eval_results['metrics'], "   Patterns: ", pattern_names)


def main():
   
    model = train_scoring_pattern(pattern_names=('P_B12',), labelled_amounts=range(100, 101), 
        sample_selection='negatives_with_none', model_name='roberta-base', 
        train_domains=['rest'], test_domains=['rest'],
        masking_strategy='aspect_scoring') #, max_steps=5, test_limit=5)    

    eval_model(scoring_model_names=["rest_100_aspect_scoring"])

if __name__ == "__main__":
    main()
