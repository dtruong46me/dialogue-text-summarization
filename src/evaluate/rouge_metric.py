import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)


def postprocess_text(preds, labels):
    nltk.download("punkt")

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric = evaluate.load("rouge")
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

if __name__=='__main__':
    pass