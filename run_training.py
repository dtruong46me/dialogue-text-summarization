import wandb
from huggingface_hub import login

import logging

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.pipelines.training_pipeline import training_pipeline
from src.utils import parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__=='__main__':
    # Load argument parser
    args = parse_args()
    logger.info("Loaded argument parser from CLI")

    checkpoint = args.checkpoint
    datapath = args.datapath
    configpath = args.configpath

    # Load token ID
    huggingface_hub_token = args.huggingface_hub_token
    wandb_token = args.wandb_token

    # Setup environment
    if huggingface_hub_token:
        os.environ["HUGGINGFACE_TOKEN"] = huggingface_hub_token
    
    if wandb_token:
        os.environ["WANDB_API_KEY"] = wandb_token

    # Login to Huggingface Hub and WandB
    login(token=huggingface_hub_token)
    wandb.login(key=wandb_token)

    training_pipeline(args)