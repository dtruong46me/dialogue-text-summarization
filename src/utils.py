import argparse

import os
import sys

import torch
import torch.nn as nn

from transformers import (
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

# from src.evaluate.rouge_metric import compute_metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tuning LLM for Dialogue Text Summarization")
    parser.add_argument("--huggingface_hub_token", type=str, default=None)
    parser.add_argument("--wandb_token", type=str, default=None)

    parser.add_argument("--checkpoint", type=str, default="google/flan-t5-base")
    parser.add_argument("--datapath", type=str, default="knkarthick/dialogsum")

    parser.add_argument("--output_dir", type=str, default="fine-tuned-flant5")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    
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

    parser.add_argument("--predict_with_generate", action="store_true")

    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--quantize", action="store_true")

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, default="q,v")
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--use_contrastive_loss", action="store_true")
    parser.add_argument("--tokenizing_strategy", type=int, default=1)

    args = parser.parse_args()
    return args


def load_training_arguments(args):
    try:
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

            predict_with_generate=args.predict_with_generate
        )

        return training_args
    
    except Exception as e:
        print(f"Error while loading training arguments: {e}")
        raise e

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, dialgue_embeddings, pos_summary_embeddings, neg_summary_embeddings):
        pos_sim = self.cosine_similarity(dialgue_embeddings, pos_summary_embeddings)
        neg_sim = self.cosine_similarity(dialgue_embeddings, neg_summary_embeddings)
        loss = torch.mean(1-pos_sim) + torch.clamp(neg_sim-self.margin, min=0.0)

        return loss

class ContrastiveLearningTrainer(Seq2SeqTrainer):
    def compute_loss(model, inputs, return_outputs=False):
        output = model(**inputs)
        lm_loss = output.loss

        dialogue_embeddings = model.encoder(inputs["input_ids"]).last_hidden_state
        pos_summary_embeddings = model.encoder(inputs["labels"]).last_hidden_state
        neg_summary_embeddings = model.encoder(inputs["negative_labels"]).last_hidden_state

        contrastive_loss = ContrastiveLoss(margin=1.0)(dialogue_embeddings, pos_summary_embeddings, neg_summary_embeddings)

        # Combine losses
        total_loss = lm_loss + contrastive_loss

        return (total_loss, output) if return_outputs else total_loss