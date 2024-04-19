
import warnings
warnings.filterwarnings("ignore")

import logging

from datasets import load_dataset

import os
import sys

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.model.models import GeneralModel
from src.evaluate.evaluation import evaluation_rouge

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluation metric")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    args = parser.parse_args()
    
    datapath = args.datapath
    checkpoint = args.checkpoint

    logger.info("Parse arguments!")

    data = load_dataset(datapath, split="test")
    logger.info(f"Loaded dataset test from: {datapath}")

    model = GeneralModel(checkpoint)
    logger.info(f"Loaded model from: {checkpoint}")

    results = evaluation_rouge(model, data)
    