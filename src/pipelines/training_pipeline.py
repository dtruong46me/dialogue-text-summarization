import os
import sys
import argparse
import numpy as np
import nltk

from nltk.tokenize import sent_tokenize
from transformers import (
    Seq2SeqTrainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from utils import *

# from model.models import load_model
from model.model import load_model
from data.preprocessing import preprocessing_data
from data.ingest_data import ingest_data

import evaluate


def training_pipeline(args: argparse.Namespace):
    try:
        print("=========================================")
        print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
        print("=========================================")

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        print(type(tokenizer))
        
        if (args.lora == False):
            print("lora=Fasle, quantize=False")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
            # model.base_model = model.get_model()
            # model.base_model.to(device)

        else:
            from peft import LoraConfig, TaskType
            from transformers import BitsAndBytesConfig
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # Define LoRA Config 
            lora_config = LoraConfig(
                r=args.lora_rank, 
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules.split(","),
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )

            if (args.quantize == True):
                print("Quantize=True, lora=True")
                model.base_model = model.prepare_quantize(bnb_config)

            if (args.quantize==False):
                print("Quantize=False, lora=True")
                model.base_model = model.get_model()
                model.base_model.to(device)

            # add LoRA adaptor
            model.base_model = model.get_peft(lora_config)
            model.base_model.print_trainable_parameters()

        # Load data from datapath
        data = ingest_data(args.datapath)
        print("\033[92mComplete loading dataset!\033[00m")

        # Pre-processing data
        data = preprocessing_data(data, tokenizer, use_contrastive_loss=args.use_contrastive_loss, tokenizing_strategy=args.tokenizing_strategy)
        print("\033[92mComplete pre-processing dataset!\033[00m")

        # Load training arguments
        training_args = load_training_arguments(args)
        print("\033[92mComplete loading training arguments!\033[00m")

        # Load metric
        metric = evaluate.load("rouge")
        nltk.download("punkt")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(sent_tokenize(label)) for label in labels]

            return preds, labels
        
        def compute_metric(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            # metric = evaluate.load("rouge")
            rouge_results = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            rouge_results = {k: round(v * 100, 4) for k, v in rouge_results.items()}
            
            results = {
                "rouge1": rouge_results["rouge1"],
                "rouge2": rouge_results["rouge2"],
                "rougeL": rouge_results["rougeL"],
                "rougeLsum": rouge_results["rougeLsum"],
                "gen_len": np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
            }

            return results
        
        # Load trainer
        if args.use_contrastive_loss==True:
            trainer = ContrastiveLearningTrainer(model=base_model,
                                     train_dataset=data["train"],
                                     eval_dataset=data["validation"],
                                     tokenizer=tokenizer,
                                     compute_metrics=compute_metric)

        if args.use_contrastive_loss==False:
            trainer = Seq2SeqTrainer(model=base_model,
                                args=training_args,
                                train_dataset=data["train"],
                                eval_dataset=data["validation"],
                                tokenizer=tokenizer,
                                compute_metrics=compute_metric)
        
        print("\033[92mComplete loading trainer!\033[00m")

        # Train model
        trainer.train()
        print("\033[92mComplete training!\033[00m")

        # Push to Huggingface Hub
        trainer.push_to_hub()
        print("\033[92mComplete pushing model to hub!\033[00m")

    except Exception as e:
        print(f"\033[31m\nError while training: {e}\033[00m")
        raise e
    