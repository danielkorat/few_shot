from utils import *

def main():
    ds_dict = load_all_datasets(train_size=200)
    # run_ds_examples(ds=ds_dict['res']['train'], model='roberta-base', pattern=P5, top_k=10)

    for domain in 'res', 'lap':
        metrics = eval_ds(ds_dict, domain, model='roberta-base', pattern=P5, top_k=10)
        print(f"Evaluation results on '{domain}' train data:\n{metrics}\n")

if __name__ == "__main__":
    main()