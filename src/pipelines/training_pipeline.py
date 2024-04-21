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

def get_trainable_parameters(model):
    """
    Returns the number of trainable parameters in the model as a string.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


def training_pipeline(args: argparse.Namespace):
    try:
        if (args.lora == False):
            # Load model from checkpoint
            model = load_model(args.checkpoint)
            logger.info("Complete loading model!")
        else:
            from peft import LoraConfig, TaskType
            from transformers import BitsAndBytesConfig
            from model.models import FlanT5Model_LoRA
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # Define LoRA Config 
            lora_config = LoraConfig(
                r=8, 
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )

            model = FlanT5Model_LoRA(args.checkpoint, bnb_config)

            if (args.quantize == True):
                # quantizing the model
                model.prepare_quantize()

            # add LoRA adaptor
            model.get_peft(lora_config)
            model.base_model.print_trainable_parameters()
            logger.info("Complete loading LoRA! " + get_trainable_parameters(model.base_model))

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
    