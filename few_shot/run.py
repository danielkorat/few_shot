from utils import load_all_datasets, evaluate, create_mlm_splits, P5
from run_pattern_mlm import main as run_pattern_mlm

def pattern_mlm_preprocess(pattern, train_size=200):
    data = load_all_datasets(train_size=train_size)

    # Prepare Pattern-MLM Training
    # Write splits to '/mlm_data'
    create_mlm_splits(data, train_size, pattern)

def train_mlm(pattern, train_domain, num_labelled, seed=42, lr=1e-05, max_seq=256, max_steps=1000,
            validation=None, model_type='roberta', model_name='roberta-base'):

    output_dir = f"p-mlm_model_{train_domain}"

    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_pattern_mlm([
    "--pattern", pattern,
    "--seed", str(seed),
    # "--num_train_epochs", "1",
    "--learning_rate", str(lr),
    "--max_seq_length", str(max_seq),
    "--max_steps", str(max_steps),
    "--train_file", "few_shot/mlm_data/" + train_domain + f'_train_{num_labelled}.txt',
    # "--validation_file", "few_shot/mlm_data/" + validation,
    # "--do_eval", "--validation_file", "few_shot/mlm_data/rest_test.txt",
    # "--evaluation_strategy", "epoch",
    "--line_by_line", "--output_dir", output_dir,
    "--model_type", model_type, "--model_name_or_path", model_name,
    "--do_train", "--overwrite_output_dir", "--overwrite_cache"])
    return output_dir

def train_eval(pattern, train_domain, num_labelled, **kwargs):
    p_mlm_model = train_mlm(pattern=pattern, train_domain=train_domain, num_labelled=num_labelled, **kwargs)
    return evaluate(lm=p_mlm_model, **dict(pattern=pattern, top_k=5))

def plot_few_shot(plot_data):
    pass

def few_shot_experiment(labelled_amounts: range(10, 101, 10), **kwargs):
    pretrained_res = evaluate(lm='roberta-base', **kwargs)
    for train_domain in 'lap', 'rest':
        plot_data = {0: pretrained_res}
        for num_labelled in labelled_amounts:
            res = train_eval(train_domain=train_domain, num_labelled=num_labelled, **kwargs)
            plot_data[num_labelled] = res
        plot_few_shot(train_domain, plot_data)

def main():
    train_eval(pattern=P5, train_domain='lap', max_steps=30)

if __name__ == "__main__":
    few_shot_experiment()
