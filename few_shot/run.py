from utils import load_all_datasets, evaluate_all, create_mlm_splits, P5
from run_pattern_mlm import main as run_pattern_mlm
import os

def eval_pretrained():
    pattern_kwargs = dict(pattern=P5, top_k=10)

    data = load_all_datasets(train_size=200)
    # run_ds_examples(data['rest']['train'], model='roberta-base', **pattern_kwargs)

    evaluate_all(data, lm='roberta-base', **pattern_kwargs)

def pattern_mlm_preprocess():
    data = load_all_datasets(train_size=200)

    # Prepare Pattern-MLM Training

    # Write splits to '/mlm_data'
    create_mlm_splits(data, P5)

def main():

    # Pattern MLM training

    # PET hparams: 
    # lr", "1x10^-5, batch_size", "16, max_len", "256, steps", "1000
    # every batch: 4 labelled + 12 unlabelled

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # run_pattern_mlm([
    #     "--pattern", P5,
    #     "--seed", "42",
    #     # "--num_train_epochs", "1",
    #     "--learning_rate", "1e-05",
    #     "--max_seq_length", "256",
    #     "--max_steps", "1000",
    #     "--validation_file", "few_shot/mlm_data/rest_test.txt",
    #     # "--do_eval", "--validation_file", "few_shot/mlm_data/rest_test.txt", "--evaluation_strategy", "epoch",
    #     "--line_by_line", "--output_dir", "pattern_mlm_model", "--train_file", "few_shot/mlm_data/lap_train.txt",
    #     "--model_type", "roberta", "--model_name_or_path", "roberta-base",
    #     "--do_train", "--overwrite_output_dir", "--overwrite_cache"])

    pattern_kwargs = dict(pattern=P5, top_k=10)
    data = load_all_datasets(train_size=200)
    evaluate_all(data, lm='pattern_mlm_model', **pattern_kwargs)    


if __name__ == "__main__":
    # main()
    eval_pretrained()
