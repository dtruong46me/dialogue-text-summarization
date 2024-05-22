import logging

import os
import sys
import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import Seq2SeqTrainer

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from utils import *

from model.models import load_model
from data.preprocessing import preprocessing_data
from data.ingest_data import ingest_data
# from evaluate.rouge_metric import compute_metrics

import evaluate

def training_pipeline(args: argparse.Namespace):
    try:
        # Load model from checkpoint
        model = load_model(args.checkpoint)
        tokenizer = model.tokenizer
        print("Complete loading model!")

        # Load data from datapath
        data = ingest_data(args.datapath)
        print("Complete loading dataset!")

        # Pre-processing data
        data = preprocessing_data(data, model.tokenizer)
        print("Complete pre-processing dataset!")

        # Load training arguments
        training_args = load_training_arguments(args)
        print("Complete loading training arguments!")

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
        trainer = Seq2SeqTrainer(model=model.base_model,
                               training_args=training_args,
                               train_dataset=data["train"],
                               eval_dataset=data["validation"],
                               tokenizer=model.tokenizer,
                               compute_metrics=compute_metric)
        print("Complete loading trainer!")

        # Train model
        trainer.train()
        print("Complete training!")

        # Push to Huggingface Hub
        trainer.push_to_hub()
        print("Complete pushing model to hub!")

    except Exception as e:
        print(f"Error while training: {e}")
        raise e
    