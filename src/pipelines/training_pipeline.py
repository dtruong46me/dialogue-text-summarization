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

        if (args.lora == True):
            from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
       
            # Define LoRA Config 
            lora_config = LoraConfig(
                r=16, 
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            if (args.quantize == True):
                # prepare int-8 model for training
                model.prepare_for_int8()

            # add LoRA adaptor
            model.get_peft(lora_config)
            model.base_model.print_trainable_parameters()
            logger.info("Complete loading LoRA! " + str(model.base_model.get_nb_trainable_parameters()))

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
                               tokenizer=model.tokenizer,
                               args=args)
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
    