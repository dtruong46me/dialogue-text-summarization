import logging

import os
import sys

from datasets import Dataset

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from model.models import load_model
from data.ingest_data import ingest_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(data_test: Dataset, model, tokenizer):
    human_summary = []
    model_summary = []

    for dialogue in dialogues:
        prompt = """Summarize the following conversation:\n{dialogue}\nSummary: """

        

if __name__=='__main__':
    datapath = "knkarthick/dialogsum"
    data = ingest_data(datapath)

    dialogues = data["test"][0:10]["dialogue"]
    summaries = data["test"][0:10]["summary"]

    print(dialogues)
    print(type(dialogues))

