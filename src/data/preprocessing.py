
from datasets import DatasetDict, Dataset
import random

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

class DialogSumDataset:
    def __init__(self, tokenizer, use_contrastive_loss=False, create_qds=False) -> None:
        self.tokenizer = tokenizer
        self.use_contrastive_loss = use_contrastive_loss
        self.create_qds = create_qds

    def handle_data(self, data: DatasetDict) -> DatasetDict:
        try:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokenized_dataset = data.map(self.preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns([key for key in data["train"][0].keys()])
            # tokenized_dataset = tokenized_dataset.filter(lambda example, index: index%100==0, with_indices=True)

            return tokenized_dataset

        except Exception as e:
            print(f"Error while tokenizing data: {e}")
            raise e
        
    def preprocess_function(self, data: Dataset) -> Dataset:
        if self.create_qds==True:
            inputs, targets = [], []
            
            checkpoint = "google/flan-t5-large"
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

            for dialogue, summary in zip(data["dialogue"], data["summary"]):
                queries = self.generate_queries(model, tokenizer, dialogue, summary, num_queries=5)
                queries = self.filter_queries(queries)

                for query in queries:
                    inputs.append(f"Query: {query} ###\nDialogue: {dialogue}")
                    targets.append(summary)
        
        if self.create_qds==False:
            prefix = "Summarize the following conversation:\n\n###"
            suffix = "\n\nSummary: "
            inputs = [prefix + input + suffix for input in data["dialogue"]]
            targets = data["summary"]

        max_source_length = 1024
        max_target_length = 176

        data["input_ids"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        # data["attention_mask"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").attention_mask
        data["labels"] = self.tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        
        # Generate negative examples:
        if self.use_contrastive_loss==True:
            negative_summaries = self.generate_negative_examples(data["summary"])
            data["negative_labels"] = self.tokenizer(negative_summaries, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            print("\033[92mComplete generate negative examples!\033[00m")

        label_ignore_ids = []
        for label in data["labels"]:
            label_example = [l if l != 0 else -100 for l in label]
            label_ignore_ids.append(label_example)

        data["labels"] = label_ignore_ids

        return data
    
    def generate_negative_examples(self, summaries):
        negative_summaries = []
        for summary in summaries:
            words = summary.split()
            random.shuffle(words)
            negative_summaries.append(" ".join(words))
        return negative_summaries

    def generate_queries(self, model, tokenizer, summary, num_queries):
        input_text = "Generate question: " + summary
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=64, num_return_sequences=num_queries, do_sample=True)
        queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return queries
    
    def filter_queries(queries):
        filtered_queries = [query for query in queries if "?" in queries and len(query.split()>3)]
        return list(set(filtered_queries))

def preprocessing_data(data: DatasetDict, tokenizer, use_contrastive_loss=False) -> DatasetDict:
    try:
        dataset_ds = DialogSumDataset(tokenizer, use_contrastive_loss)
        tokenized_data = dataset_ds.handle_data(data)

        return tokenized_data

    except Exception as e:
        print(f"Error while pre-processing data: {e}")
        raise e