import logging
from datasets import load_dataset
from datasets import DatasetDict

class IngestDataset:
    def __init__(self, datapath: str="knkarthick/dialogsum") -> None:
        self.datapath = datapath

    def get_data(self) -> DatasetDict:
        print(f"Loading data from {self.datapath}")

        data = load_dataset(self.datapath, trust_remote_code=True)
        return data
    

def ingest_data(datapath: str) -> DatasetDict:
    try:
        ingest_data = IngestDataset(datapath)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        print(f"\nError while loading data: {e}")
        raise e