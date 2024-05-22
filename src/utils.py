import argparse

import os
import sys
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import yaml

import logging

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

# from src.evaluate.rouge_metric import compute_metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tuning LLM for Dialogue Text Summarization")
    parser.add_argument("--configpath", type=str, default=None)
    parser.add_argument("--huggingface_hub_token", type=str, default=None)
    parser.add_argument("--wandb_token", type=str, default=None)

    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")

    parser.add_argument("--output_dir", type=str, default="fine-tuned-flant5")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--weight_decay", type=float, default=0.005)

    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--save_strategy", type=str, default="no")

    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=1)

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="flan-t5-base-model")

    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--load_best_model_at_end", type=bool, default=False)

    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--sortish_sampler", type=bool, default=True)
    parser.add_argument("--predict_with_generate", type=bool, default=True)
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
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,

                evaluation_strategy=args.evaluation_strategy,
                save_strategy=args.save_strategy,
                
                logging_strategy=args.logging_strategy,
                logging_steps=args.logging_steps,
                save_total_limit=args.save_total_limit,
                
                report_to=args.report_to,
                run_name=args.run_name,

                metric_for_best_model=args.metric_for_best_model,
                load_best_model_at_end=args.load_best_model_at_end,

                fp16=args.fp16,
                sortish_sampler=args.sortish_sampler,
                predict_with_generate=args.predict_with_generate
            )

        return training_args
    
    except Exception as e:
        print(f"Error while loading training arguments: {e}")
        raise e

def load_callbacks(args) -> list:
    try:
        callbacks = []
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping_callback)
        return callbacks
    
    except Exception as e:
        print(f"Error while loading callbacks: {e}")
        raise e

def load_trainer(model, training_args, dataset, tokenizer, args):
    try:
        # callbacks = load_callbacks(args)
        # def custom_compute_metrics(eval_preds):
        #     metrics = compute_metrics(eval_preds, tokenizer)

            # wandb.log(metrics)

            # return metrics

        # callbacks = [WandBCallback(tokenizer)]

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            # callbacks=callbacks,
            # compute_metrics=custom_compute_metrics
        )
        
        return trainer
    
    except Exception as e:
        print(f"Error while loading trainer: {e}")
        raise e

def load_config(configpath):
    if os.path.exists(configpath):
        with open(configpath, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    else:
        return None

# if __name__=='__main__':
#     args = parse_args()
#     print(args)