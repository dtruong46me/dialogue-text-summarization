
import os, sys

import argparse

from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import login

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

def merge_dataset(datapaths) -> Dataset:
    dataset = load_dataset(datapaths[0], split="train")

    for i in range(1, len(datapaths)):
        data = load_dataset(datapaths[i], split="train")
        data = concatenate_datasets([dataset, data])
    
    return dataset


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapaths", type=str, default="")
    parser.add_argument("--huggingface_hub_token", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    print("=========================================")
    print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
    print("=========================================")

    login(token=args.huggingface_hub_token)
    print("Successfully logged in to Huggingface Hub")

    dataset = merge_dataset(datapaths=args.datapaths)
    
    DATASET_ID = "qds-triplet-dialogsum"
    dataset.push_to_hub(DATASET_ID)
    print(f"Successful push to Huggingface Hub: {DATASET_ID}")