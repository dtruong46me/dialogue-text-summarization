import wandb
from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore")

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.pipelines.training_pipeline import training_pipeline
from src.utils import parse_args

def main():
    # Load argument parser
    args = parse_args()
    print(f"\033[92mLoaded argument parsers\033[00m")

    # Load token ID
    huggingface_hub_token = args.huggingface_hub_token
    wandb_token = args.wandb_token

    if wandb_token:
        os.environ["WANDB_PROJECT"] = "nlp_project"

    # Login to Huggingface Hub and WandB
    login(token=huggingface_hub_token)
    print("\033[92mSuccessful login to Huggingface Hub\033[00m")
    
    wandb.login(key=wandb_token)
    print("\033[92mSuccessful login to WandB\033[00m")

    training_pipeline(args)
    print("\033[92mFinish training pipeline\033[00m")

if __name__=='__main__':
    main()