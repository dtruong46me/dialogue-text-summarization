import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset

import os
import sys

import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.model.models import GeneralModel
from src.evaluate.evaluation import evaluation_rouge
from transformers import GenerationConfig

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluation metric")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    parser.add_argument("--resultpath", type=str, default="./")
    
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

    model = GeneralModel(checkpoint)
    print(f"Loaded model from: {checkpoint}")

    results = evaluation_rouge(model, data)
    print(results)