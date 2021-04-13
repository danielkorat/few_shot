from utils import load_all_datasets, evaluate, create_mlm_train_sets, plot_few_shot, PATTERNS
from run_pattern_mlm import main as run_pattern_mlm
import os
import pickle
import time

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def pattern_mlm_preprocess(labelled_amounts, sample_selection, **kwargs):
    datasets = load_all_datasets()
    res = {}
    # Prepare Pattern-MLM Training
    # Write splits to '/mlm_data'
    for num_labelled in labelled_amounts:
        amounts = create_mlm_train_sets(datasets, num_labelled, sample_selection, **kwargs)
        res[num_labelled] = amounts
    return res

def train_mlm(train_domain, num_labelled, pattern_names, seed=42, lr=1e-05, max_seq=256, max_steps=1000, batch_size=16,
            validation=None, model_type='roberta', model_name='roberta-base', **kwargs):
    hparams = locals()
    for v in 'train_domain', 'num_labelled', 'kwargs':
        hparams.pop(v)

    os.makedirs('models', exist_ok=True)
    output_dir = f"models/p-mlm_model_{train_domain}_{num_labelled}"

    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_pattern_mlm([
    "--pattern", PATTERNS[pattern_names[0]],
    "--seed", str(seed),
    # "--num_train_epochs", "1",
    "--learning_rate", str(lr),
    "--max_seq_length", str(max_seq),
    "--max_steps", str(max_steps),
    "--train_file", "mlm_data/" + train_domain + f'_train_{num_labelled}_{pattern_names[0]}.txt',
    "--per_device_train_batch_size", str(batch_size),
    # "--validation_file", "mlm_data/" + validation,
    # "--do_eval", "--validation_file", "mlm_data/rest_test.txt",
    # "--evaluation_strategy", "epoch",
    "--line_by_line", "--output_dir", output_dir,
    "--model_type", model_type, "--model_name_or_path", model_name,
    "--do_train", "--overwrite_output_dir", "--overwrite_cache"])
    return output_dir, hparams

def train_eval(train_domain, num_labelled, **kwargs):
    p_mlm_model, hparams = train_mlm(train_domain, num_labelled, **kwargs)
    kwargs['model_name'] = p_mlm_model
    eval_res = evaluate(exper_name=f"{num_labelled}", **kwargs)
    return eval_res, hparams

def few_shot_experiment(labelled_amounts, **kwargs):
    actual_num_labelled = pattern_mlm_preprocess(labelled_amounts, **kwargs)
    pretrained_res = evaluate(**kwargs)
    for train_domain in kwargs['train_domains']:
        print(f"\n{'=' * 50}\n\t\t  Train Domain: {train_domain}\n{'=' * 50}")
        plot_data = {0: pretrained_res}
        for num_labelled in labelled_amounts:
            print(f"\n{'-' * 50}\n\t\t  Num. Labelled: {num_labelled}\n{'-' * 50}")
            res, train_hparams = train_eval(train_domain, num_labelled, **kwargs)
            plot_data[num_labelled] = res
        with open(f'plots/{train_domain}_plot_data_{time.strftime("%Y%m%d-%H%M%S")}.pkl', 'wb') as f:
            pickle.dump((plot_data, train_hparams, actual_num_labelled), f)
        plot_few_shot(train_domain, plot_data, train_hparams, actual_num_labelled, **kwargs)


def evaluate_patterns(pattern_names_list=(['P1', 'P2'], ['P2']), **kwargs):
    for pattern_names in pattern_names_list:
        evaluate(pattern_names=pattern_names, **kwargs)


def main(smoke=False):
    few_shot_experiment(
        pattern_names=('P5',),
        labelled_amounts=range(100, 101),
        sample_selection='negatives_with_none',
        model_name='roberta-base',
        train_domains=['rest'],
        test_domains=['rest'],
        max_steps=5 if smoke else 1000,
        test_limit=5 if smoke else None
        )

if __name__ == "__main__":
    
    main()

    main(smoke=True)

    # lap_plot = 'plots/lap_plot_data_20210404-175319.pkl'
    # rest_plot = 'plots/rest_plot_data_20210404-183737.pkl'

    # plot_few_shot('lap', *pickle.load(open(lap_plot, 'rb')))
    # plot_few_shot('rest', *pickle.load(open(rest_plot, 'rb')))
