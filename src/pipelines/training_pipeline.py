import logging

import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from utils import *

from model.models import load_model
from data.preprocessing import preprocessing_data
from data.ingest_data import ingest_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def training_pipeline(args: argparse.Namespace):
    try:
        # Load model from checkpoint
        model = load_model(args.checkpoint)
        logger.info("Complete loading model!")

        # Load data from datapath
        data = ingest_data(args.datapath)
        logger.info("Complete loading dataset!")

        # Pre-processing data
        data = preprocessing_data(data, model.tokenizer)
        logger.info("Complete pre-processing dataset!")

        # Load training arguments
        training_args = load_training_arguments(args)
        logger.info("Complete loading training arguments!")

        # Load trainer
        trainer = load_trainer(model=model.base_model,
                               training_args=training_args,
                               dataset=data,
                               tokenizer=model.tokenizer)
        logger.info("Complete loading trainer!")

        # Train model
        trainer.train()
        logger.info("Complete training!")

        # Push to Huggingface Hub
        trainer.push_to_hub()
        logger.info("Complete pushing model to hub!")

    except Exception as e:
        logger.error(f"Error while training: {e}")
        raise e
    