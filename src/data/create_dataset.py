
import sys, os

import argparse

from bert_score import BERTScorer

from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoTokenizer
)

import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import login

from datasets import load_dataset, Dataset

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from preprocessing import *

def create_qds_triplet(datapath, split, start_index, end_index) -> Dataset:
    data = load_dataset(datapath, split=split)
    data = Dataset.from_dict(data[start_index:end_index])

    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    CHECKPOINT = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT)
    model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT)

    qds_triplet = {
        "query": [],
        "dialogue": [],
        "summary": []
    }

    dsp = DialogSumDataset(
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    )

    for dialogue, summary in zip(data["dialogue"], data["summary"]):
        answerable_queries = []

        while len(answerable_queries) < 1:
            queries = dsp.generate_queries(model, tokenizer, summary, num_queries=5)

            for query in queries:
                ## Text based filtering
                output = dsp.text_based_filtering(model, tokenizer, query, summary)
                if "yes" in output.lower():
                    answerable_queries.append(query)

        n = len(answerable_queries)
        print("Length of answerable queries:", n, end="  ###  ")

        if n == 1:
            qds_triplet["query"].append(answerable_queries[0])
            qds_triplet["dialogue"].append(dialogue)
            qds_triplet["summary"].append(summary)

        if n > 1:
            filtered_queries = []
            scores = [[0.0]*n for _ in range(n)]

            for i in range(n):
                for j in range(n):
                    if i > j:
                        scores[i][j] = qds_triplet.semantic_filtering(scorer, answerable_queries[i], answerable_queries[j])
            
            keep_indices = set(range(n))
            for i in range(n):
                for j in range(n):
                    if scores[i][j] > 0.7 and i > j:
                        keep_indices.discard(j)
            
            for i in sorted(keep_indices):
                filtered_queries.append(answerable_queries[i])
            
            print("Length of filtered queries:", len(filtered_queries), end="  ###  ")

            for query in filtered_queries:
                qds_triplet["query"].append(query)
                qds_triplet["dialogue"].append(dialogue)
                qds_triplet["summary"].append(summary)

        print("Length of inputs:", len(qds_triplet["summary"]))

    return Dataset.from_dict(qds_triplet)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--huggingface_hub_token", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()

    print("=========================================")
    print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
    print("=========================================")

    login(token=args.huggingface_hub_token)
    print("Successfully logged in to Huggingface Hub")

    qds_triplet = create_qds_triplet(args.datapath, args.split, args.start_index, args.end_index)

    save_name = f"dialogsum-{args.split}-{args.start_index}-{args.end_index}"
    qds_triplet.push_to_hub(save_name)
    print(f"Saved to: {save_name}")