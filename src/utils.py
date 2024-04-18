import argparse

import os
import sys
import yaml

import logging

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from src.evaluate.rouge_metric import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tuning LLM for Dialogue Text Summarization")
    parser.add_argument("--configpath", type=str, default=None)
    parser.add_argument("--huggingface_hub_token", type=str, default=None)
    parser.add_argument("--wandb_token", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--output_dir", type=str, default="fine-tuned-flant5-base")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="flan-t5-base-model")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    args = parser.parse_args()
    return args

def load_training_arguments(args):
    try:
        if args.configpath is not None:
            config = load_config(configpath=args.configpath)
            training_args = Seq2SeqTrainingArguments(**config["training_args"])

        else:
            training_args = Seq2SeqTrainingArguments(
                output_dir=args.output_dir,
                overwrite_output_dir=args.overwrite_output_dir,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                weight_decay=args.weight_decay,
                evaluation_strategy=args.evaluation_strategy,
                logging_strategy=args.logging_strategy,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                save_strategy=args.save_strategy,
                hub_strategy=args.hub_strategy,
                logging_steps=args.logging_steps,
                save_total_limit=args.save_total_limit,
                learning_rate=args.learning_rate,
                push_to_hub=args.push_to_hub,
                report_to=args.report_to,
                metric_for_best_model=args.metric_for_best_model,
                run_name=args.run_name
            )

        return training_args
    
    except Exception as e:
        logger.error(f"Error while loading training arguments: {e}")
        raise e

def load_callbacks() -> list:
    try:
        callbacks = []
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=2
        )
        callbacks.append(early_stopping_callback)
        return callbacks
    
    except Exception as e:
        logger.error(f"Error while loading callbacks: {e}")
        raise e

def load_trainer(model, training_args, dataset, tokenizer):
    try:
        callbacks = load_callbacks(args)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics(tokenizer=tokenizer)
        )
        return trainer
    
    except Exception as e:
        logger.error(f"Error while loading trainer: {e}")
        raise e

def load_config(configpath):
    if os.path.exists(configpath):
        with open(configpath, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    else:
        return None

if __name__=='__main__':
    args = parse_args()
    print(args)