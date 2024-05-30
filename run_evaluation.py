import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset

import os, sys

import pandas as pd
import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.model.model import load_model
from src.evaluate.evaluation import evaluation_rouge
from transformers import GenerationConfig


def save_metrics_to_csv(results, resultpath, checkpoint):
    
    results["checkpoint"] = checkpoint

    # Convert results to DataFrame
    df = pd.DataFrame([results])

    if not os.path.isfile(resultpath):
        df.to_csv(resultpath, index=False)
    else:
        df.to_csv(resultpath, mode='a', header=False, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluation metric")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    parser.add_argument("--resultpath", type=str, default="results/rouge_score.csv")
    
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()
    
    datapath = args.datapath
    checkpoint = args.checkpoint

    generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )

    data = load_dataset(datapath, split="test")

    model = load_model(checkpoint)
    print(f"Loaded model from: {checkpoint}")

    results = evaluation_rouge(model, data, generation_config)
    
    print("--------------------------")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("--------------------------")

    save_metrics_to_csv(results, args.resultpath, checkpoint)
    print(f"Results saved to: {args.resultpath}")

if __name__ == "__main__":
    main()