from utils import *

def eval_pretrained():
    pattern_kwargs = dict(pattern=P5, top_k=10)

    data = load_all_datasets(train_size=200)
    # run_ds_examples(ds=ds_dict['rest']['train'], model='roberta-base', **pattern_kwargs)

    evaluate_all(data, lm='roberta-base', **pattern_kwargs)

def pattern_mlm_preprocess():
    data = load_all_datasets(train_size=200)

    # Prepare Pattern-MLM Training

    # Write splits to '/mlm_data'
    create_mlm_splits(data, P5)

def main():
    
    # Pattern MLM training

    # PET hparams: 
    # lr=1x10^-5, batch_size=16, max_len=256, steps=1000
    # every batch: 4 labelled + 12 unlabelled

    # Run this training command in shell:
    # python run_pattern_mlm.py --seed=42 --num_train_epochs=1 --learning_rate=5e-04 --line_by_line \
    #     --output_dir=pattern_mlm --train_file=data/lap_train.txt --validation_file=data/rest_test.txt --per_device_train_batch_size=1 \
    #     --model_type=roberta --model_name_or_path=roberta-base --do_train --do_eval \
    #     --overwrite_output_dir --overwrite_cache --evaluation_strategy=epoch

    # evaluate_all(data, model='pattern_mlm', **pattern_kwargs)    


if __name__ == "__main__":
    main()
