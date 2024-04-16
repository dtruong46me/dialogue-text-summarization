import logging
from abc import ABC, abstractclassmethod

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from ingest_data import ingest_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStrategy(ABC):
    """
    Abstract class for handling data
    """
    @abstractclassmethod
    def handle_data(self, data, *args) -> None:
        pass


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: Dataset, *args) -> DatasetDict:
        """
        If loaded dataset is not divided, this method is used to split data using 
        Dataset.train_test_split() of hugging face
        """
        try:
            pass

        except Exception as e:
            logger.error(f"Error while dividing data: {e}")
            raise e
        

class DataTokenizingStrategy(DataStrategy):
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
            logger.info("Error while tokenizing data: {e}")
            raise e
        
    def preprocess_function(self, data: Dataset, *args) -> Dataset:
        prefix = "Summarize the following conversation:\\nn"
        suffix = "\n\nSummary: "
        inputs = [prefix + input + suffix for input in data["dialogue"]]

        data["input_ids"] = self.tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt").input_ids
        data["attention_mask"] = self.tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt").attention_mask
        data["labels"] = self.tokenizer(data["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

        label_ignore_ids = []
        for label in data["labels"]:
            label_example = [l if l != 0 else -100 for l in label]
            label_ignore_ids.append(label_example)

        data["labels"] = label_ignore_ids

        return data

# if __name__=='__main__':
#     checkpoint = "google/flan-t5-base"
#     datapath = "knkarthick/dialogsum"

#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#     dataset = ingest_data(datapath)
#     print(dataset["train"])

#     print("\n\n\n")
#     tokenizing_data = DataTokenizingStrategy(tokenizer)
#     tokenized_dataset = tokenizing_data.handle_data(dataset)

#     print(tokenized_dataset)

#     print(type(tokenized_dataset))

#     print(type(tokenized_dataset["train"]))

#     print(tokenized_dataset["train"][0])