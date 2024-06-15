
from datasets import DatasetDict, Dataset
import random
from bert_score import BERTScorer

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

class DialogSumDataset:
    def __init__(self, tokenizer, use_contrastive_loss=False, tokenizing_strategy=1) -> None:
        self.tokenizer = tokenizer
        self.use_contrastive_loss = use_contrastive_loss
        self.tokenizing_strategy = tokenizing_strategy
        
    def handle_data(self, data: DatasetDict) -> DatasetDict:
        try:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokenized_dataset = data.map(self.preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns([key for key in data["train"][0].keys()])

            print("+++++++++++++++++++")
            print(tokenized_dataset)
            print("+++++++++++++++++++")
            
            return tokenized_dataset

        except Exception as e:
            print(f"\033[31m\nError while tokenizing data: {e}\033[00m")
            raise e

    def preprocess_function(self, data: Dataset) -> Dataset:
        ###
        if self.tokenizing_strategy<=2:
            prefix = "Summarize the following conversation:\n###\n"
            suffix = "\n###\nSummary: "
            inputs = [prefix + input + suffix for input in data["dialogue"]]
            targets = data["summary"]
            
            if self.tokenizing_strategy==1:
                max_source_length = 1024
                max_target_length = 176

            if self.tokenizing_strategy==2:
                max_source_length = 1224
                max_target_length = 176

        if self.tokenizing_strategy==3:
            inputs = ["### Instruction: " + instruction + "\n### Input: " + input + "\n### Response: " for instruction, input in zip(data["instruction"], data["input"])]
            targets = data["output"]

            max_source_length = 1224
            max_target_length = 176

        print("- Max source length: ", max_source_length)
        print("- Max target length: ", max_target_length)

        data["input_ids"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        # data["attention_mask"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").attention_mask
        data["labels"] = self.tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        
        # Generate negative examples:
        if self.use_contrastive_loss==True:
            negative_summaries = self.generate_negative_examples(data["summary"])
            data["negative_labels"] = self.tokenizer(negative_summaries, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            print("Complete generate negative examples!")

        label_ignore_ids = []
        for label in data["labels"]:
            label_example = [l if l != 0 else -100 for l in label]
            label_ignore_ids.append(label_example)

        data["labels"] = label_ignore_ids

        return data
    
    ## Create Negetive Example for Contrastive Learning
    def generate_negative_examples(self, summaries):
        negative_summaries = []
        for summary in summaries:
            words = summary.split()
            random.shuffle(words)
            negative_summaries.append(" ".join(words))
        return negative_summaries

    ## Create Instruction Dataset
    def generate_queries(self, model, tokenizer, summary, num_queries):
        input_text = "Generate an answerable and specific question based on the following context:. ###\nContext: " + summary
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=64, num_return_sequences=num_queries, do_sample=True)
        queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return queries
    
    def text_based_filtering(self, model, tokenizer, query, summary):
        input_text = "Is the question fully answerable from the context without any guessing, yes or no?###\nQuestion: " + query + "###\nContext: " + summary + "###Answer: "
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, num_return_sequences=1)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    
    def semantic_filtering(self, scorer, query1, query2):
        score = scorer.score([query1], [query2])[0]
        return score


def preprocessing_data(data: DatasetDict, tokenizer, use_contrastive_loss=False, tokenizing_strategy=False) -> DatasetDict:
    try:
        dataset_ds = DialogSumDataset(tokenizer, use_contrastive_loss, tokenizing_strategy)
        tokenized_data = dataset_ds.handle_data(data)

        return tokenized_data

    except Exception as e:
        print(f"\nError while pre-processing data: {e}")
        raise e