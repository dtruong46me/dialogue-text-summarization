import logging

import os
import sys

from datasets import DatasetDict

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from data_strategy import *
from ingest_data import *

# checkpoint = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class DataPreprocessing:
    def __init__(self, data: DatasetDict, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self, *args):
        try:
            return self.strategy.handle_data(self.data, *args)
        
        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise e
        

def preprocessing_data(data: DatasetDict, tokenizer, *args) -> DatasetDict:
    try:
        tokenizing_strategy = DataTokenizingStrategy(tokenizer)
        data_preprocess = DataPreprocessing(data, tokenizing_strategy)

        tokenized_data = data_preprocess.handle_data()

        return tokenized_data

    except Exception as e:
        logger.error(f"Error while pre-processing data: {e}")
        raise e
    
# if __name__=='__main__':
#     checkpoint = "google/flan-t5-base"
#     datapath = "knkarthick/dialogsum"

#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
#     data = ingest_data(datapath)

#     data = preprocessing_data(data, tokenizer)

#     print(data)