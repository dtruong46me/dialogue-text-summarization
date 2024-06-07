import os
import sys

from datasets import Dataset

import evaluate
import torch

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from model.model import Model


class RougeEvaluation:
    def __init__(self) -> None:
        self.rouge_metric = evaluate.load("rouge")
        
    def compute_rouge_metric(self, generated_summary, reference_summary) -> dict:
        results = self.rouge_metric.compute(
            predictions=generated_summary,
            references=reference_summary,
            use_aggregator=True,
            use_stemmer=True
        )
        return results
    

def evaluation_rouge(model: Model, data: Dataset, generation_config) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.base_model = model.get_model()
    model.base_model.to(device)

    dialogues = data["dialogue"]

    human_summaries = [summary for summary in data["summary"]]

    model_summaries = []

    prefix = "Summarize the following dialogue:\n###"
    suffix = "\n###Summary: "

    for idx, dialogue in enumerate(dialogues):
        input = prefix + dialogue + "\n###The generated summary should be around " + str(int(0.2*len(dialogue.split()))) + " words." + suffix
        
        print(idx, end="# ")
        output_text = model.generate_summary(input, generation_config, do_sample=False)

        model_summaries.append(output_text)

    rouge_evaluator = RougeEvaluation()

    results = rouge_evaluator.compute_rouge_metric(model_summaries, human_summaries)

    generated_lengths = [len(summary.split()) for summary in model_summaries]
    average_gen_len = sum(generated_lengths) / len(generated_lengths) if generated_lengths else 0

    results["gen_len"] = average_gen_len
    
    return results