from utils import *

def main():
    (res_train, res_test), (lap_train, lap_test) = load_all_datasets(train_size=200)
    # fm_pipeline = get_fm_pipeline('roberta-base')
    run_ds_examples(ds=res_train, model='roberta-base', pattern=P5, top_k=10)

if __name__ == "__main__":
    main()
