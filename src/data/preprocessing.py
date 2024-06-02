
from datasets import DatasetDict, Dataset
import random

class DataTokenizing:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def handle_data(self, data: DatasetDict, *args) -> DatasetDict:
        try:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokenized_dataset = data.map(self.preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns([key for key in data["train"][0].keys()])
            # tokenized_dataset = tokenized_dataset.filter(lambda example, index: index%100==0, with_indices=True)

            return tokenized_dataset

        except Exception as e:
            print(f"Error while tokenizing data: {e}")
            raise e
        
    def preprocess_function(self, data: Dataset, generate_negatievs=False) -> Dataset:
        prefix = "Summarize the following conversation:\n\n###"
        suffix = "\n\nSummary: "
        inputs = [prefix + input + suffix for input in data["dialogue"]]

        max_source_length = 1024
        max_target_length = 176

        data["input_ids"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        # data["attention_mask"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").attention_mask
        data["labels"] = self.tokenizer(data["summary"], max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        
        # Generate negative examples:
        if generate_negatievs==True:
            negative_summaries = self.generate_negative_examples(data["summary"])
            data["negative_labels"] = self.tokenizer(negative_summaries, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

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
    

def preprocessing_data(data: DatasetDict, tokenizer) -> DatasetDict:
    try:
        tokenizing_data = DataTokenizing(tokenizer)
        tokenized_data = tokenizing_data.handle_data(data)

        return tokenized_data

    except Exception as e:
        print(f"Error while pre-processing data: {e}")
        raise e