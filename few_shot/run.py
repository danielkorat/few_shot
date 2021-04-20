from patterns import ROOT
from utils import load_all_datasets, evaluate, create_mlm_train_sets, plot_few_shot, PATTERNS
from run_pattern_mlm import main as run_pattern_mlm
import os
import pickle
import time

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def pattern_mlm_preprocess(num_labelled_list, train_domains, sample_selection, **kwargs):
    ds = load_all_datasets()
    res = {}
    # Prepare Pattern-MLM Training
    # Write splits to '/mlm_data'
    for num_labelled in num_labelled_list:
        amounts = create_mlm_train_sets(datasets=ds, num_labelled=num_labelled,
            sample_selection=sample_selection, train_domains=train_domains, **kwargs)
        res[num_labelled] = amounts
    return res

def train_mlm(train_domain, num_labelled, pattern_names, sample_selection, seed=42, lr=1e-05, max_seq=256, max_steps=1000, batch_size=16,
            validation=None, model_type='roberta', model_name='roberta-base', **kwargs):
    hparams = locals()
    for v in 'train_domain', 'num_labelled', 'kwargs':
        hparams.pop(v)
    trained_model_names = []
    os.makedirs('models', exist_ok=True)

    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for pattern_name in pattern_names:
        print(f"Running train_mlm() for pattern {pattern_name}...")
        exper_str = f"{train_domain}_{pattern_name}_{num_labelled}_{sample_selection}"
        trained_model_names.append(exper_str)

        run_pattern_mlm([
        "--seed", str(seed),
        "--model_type", model_type, 
        "--pattern", PATTERNS[pattern_name],
        # "--num_train_epochs", "1",
        "--learning_rate", str(lr),
        "--max_seq_length", str(max_seq),
        "--max_steps", str(max_steps),
        "--train_file", str(ROOT / "mlm_data" / f"{exper_str}.txt"),
        "--per_device_train_batch_size", str(batch_size),
        "--line_by_line", 
        "--output_dir", str(ROOT / "models" f"{exper_str}"),
        "--do_train", "--overwrite_output_dir", 
        "--model_name_or_path", model_name,
        "--overwrite_cache"
        # "--validation_file", "mlm_data/" + validation,
        # "--do_eval", "--validation_file", "mlm_data/rest_test.txt",
        # "--evaluation_strategy", "epoch",
        ])

    return trained_model_names, hparams

def train_eval(train_domain, num_labelled, **kwargs):
    trained_models, hparams = train_mlm(train_domain, num_labelled, **kwargs)
    kwargs.pop('model_names')
    eval_res = evaluate(model_names=trained_models, **kwargs)
    return eval_res, hparams


def few_shot_experiment(num_labelled_list, train_domains, **kwargs):
    actual_num_labelled_list = pattern_mlm_preprocess(num_labelled_list, train_domains, **kwargs)
    pretrained_res = evaluate(**kwargs)
    for train_domain in train_domains:
        print(f"Running few_shot_experiment() for train domain {train_domain}...")
        print(f"\n{'=' * 50}\n\t\t  Train Domain: {train_domain}\n{'=' * 50}")
        plot_data = {0: pretrained_res}
        for num_labelled in num_labelled_list:
            print(f"Running num_labelled {num_labelled}...")
            print(f"\n{'-' * 50}\n\t\t  Num. Labelled: {num_labelled}\n{'-' * 50}")
            res, train_hparams = train_eval(train_domain, num_labelled, **kwargs)
            plot_data[num_labelled] = res
        with open(str(ROOT / 'plots' / f'{train_domain}_plot_data_{time.strftime("%Y%m%d-%H%M%S")}.pkl'), 'wb') as f:
            pickle.dump((plot_data, train_hparams, actual_num_labelled_list), f)
        plot_few_shot(train_domain, plot_data, train_hparams, actual_num_labelled_list, **kwargs)


def main(smoke=False):
    few_shot_experiment(
        # pattern_names=('P1',), # 'P2', 'P3', 'P4', 'P5', 'P6'),
        pattern_names=[f'P{i}' for i in range(1, 10)],
        # num_labelled_list=range(50, 151, 50),
        num_labelled_list=range(100, 101),
        sample_selection='negatives_with_none',
        model_names=('roberta-base',),
        train_domains=['rest'],
        test_domains=['rest'],
        max_steps=5 if smoke else 1000,
        test_limit=5 if smoke else None
        )

if __name__ == "__main__":

    # main()

    main(smoke=True)


    # lap_plot = 'plots/lap_plot_data_20210404-175319.pkl'
    # rest_plot = 'plots/rest_plot_data_20210404-183737.pkl'

    # plot_few_shot('lap', *pickle.load(open(lap_plot, 'rb')))
    # plot_few_shot('rest', *pickle.load(open(rest_plot, 'rb')))
