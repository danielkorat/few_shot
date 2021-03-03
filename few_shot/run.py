from utils import load_all_datasets, evaluate, create_mlm_train_sets, PATTERNS
from run_pattern_mlm import main as run_pattern_mlm

def pattern_mlm_preprocess(labelled_amounts, pattern_name):
    datasets = load_all_datasets()
    res = {}
    # Prepare Pattern-MLM Training
    # Write splits to '/mlm_data'
    for num_labelled in labelled_amounts:
        amounts = create_mlm_train_sets(datasets, num_labelled, pattern_name)
        res[num_labelled] = amounts
    return res

def train_mlm(train_domain, num_labelled, pattern_name, seed=42, lr=1e-05, max_seq=256, max_steps=1000, batch_size=16,
            validation=None, model_type='roberta', model_name='roberta-base'):

    output_dir = f"p-mlm_model_{train_domain}"

    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_pattern_mlm([
    "--pattern", PATTERNS[pattern_name],
    "--seed", str(seed),
    # "--num_train_epochs", "1",
    "--learning_rate", str(lr),
    "--max_seq_length", str(max_seq),
    "--max_steps", str(max_steps),
    "--train_file", "few_shot/mlm_data/" + train_domain + f'_train_{num_labelled}_{pattern_name}.txt',
    "--per_device_train_batch_size", str(batch_size),
    # "--validation_file", "few_shot/mlm_data/" + validation,
    # "--do_eval", "--validation_file", "few_shot/mlm_data/rest_test.txt",
    # "--evaluation_strategy", "epoch",
    "--line_by_line", "--output_dir", output_dir,
    "--model_type", model_type, "--model_name_or_path", model_name,
    "--do_train", "--overwrite_output_dir", "--overwrite_cache"])
    return output_dir

def train_eval(train_domain, num_labelled, **kwargs):
    p_mlm_model = train_mlm(train_domain=train_domain, num_labelled=num_labelled, **kwargs)
    return evaluate(lm=p_mlm_model, exper_name=f"{num_labelled}", **kwargs)

def plot_few_shot(plot_data):
    print(plot_data)

def few_shot_experiment(labelled_amounts, **kwargs):
    actual_num_labelled = pattern_mlm_preprocess(labelled_amounts, kwargs['pattern_name'])
    pretrained_res = evaluate(lm='roberta-base', **kwargs)
    for train_domain in 'lap', 'rest':
        plot_data = {0: pretrained_res}
        for num_labelled in labelled_amounts:
            res = train_eval(train_domain=train_domain, num_labelled=num_labelled, **kwargs)
            plot_data[num_labelled] = res
        plot_few_shot(train_domain, actual_num_labelled, plot_data)

def main():
    few_shot_experiment(pattern_name='P5', labelled_amounts=range(10, 41, 10), max_steps=50)

if __name__ == "__main__":
    main()
