import wandb
from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore")

import logging

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.pipelines.training_pipeline import training_pipeline
from src.utils import parse_args

def main():
    # Load argument parser
    args = parse_args()
    print("Loaded argument parsers")

    checkpoint = args.checkpoint
    datapath = args.datapath

    # Load token ID
    huggingface_hub_token = args.huggingface_hub_token
    wandb_token = args.wandb_token

    # Setup environment
    if huggingface_hub_token:
        os.environ["HUGGINGFACE_TOKEN"] = huggingface_hub_token
    
    if wandb_token:
        os.environ["WANDB_PROJECT"] = "nlp_project"

    # Login to Huggingface Hub and WandB
    login(token=huggingface_hub_token)
    print("Successful login to Huggingface Hub")
    wandb.login(key=wandb_token)
    print("Successful login to WandB")

    training_pipeline(args)
    print("Finish training pipeline")

if __name__=='__main__':
    main()