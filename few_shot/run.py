from utils import load_all_datasets, evaluate_all, create_mlm_splits, P5
from run_pattern_mlm import main as run_pattern_mlm
import os

def eval_pretrained():
    pattern_kwargs = dict(pattern=P5, top_k=5)
    # run_ds_examples(data['rest']['train'], model='roberta-base', **pattern_kwargs)
    evaluate_all(lm='roberta-base', **pattern_kwargs)

def pattern_mlm_preprocess():
    data = load_all_datasets(train_size=200)

    # Prepare Pattern-MLM Training

    # Write splits to '/mlm_data'
    create_mlm_splits(data, P5)

def train_mlm(pattern, train_domain, seed=42, lr=1e-05, max_seq=256, max_steps=1000,
            validation=None, model_type='roerta', model_name='roberta-base'):

    # hparams used in PET: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled

    output_dir = f"p-mlm_model_{train_domain}"

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_pattern_mlm([
    "--pattern", pattern,
    "--seed", str(seed),
    # "--num_train_epochs", "1",
    "--learning_rate", lr,
    "--max_seq_length", max_seq,
    "--max_steps", max_steps,
    "--train_file", "few_shot/mlm_data/" + train_domain + '._train.txt',
    # "--validation_file", "few_shot/mlm_data/" + validation,
    # "--do_eval", "--validation_file", "few_shot/mlm_data/rest_test.txt",
    # "--evaluation_strategy", "epoch",
    "--line_by_line", "--output_dir", output_dir,
    "--model_type", "roberta", "--model_name_or_path", "roberta-base",
    "--do_train", "--overwrite_output_dir", "--overwrite_cache"])

    return output_dir

def main():
    p_mlm_model = train_mlm(pattern = P5, train_domain='rest')
    evaluate_all(lm=p_mlm_model, **dict(pattern=P5, top_k=5))    


if __name__ == "__main__":
    main()
    # eval_pretrained()
