import logging
from datasets import load_dataset
from datasets import DatasetDict

class IngestDataset:
    def __init__(self, datapath: str="knkarthick/dialogsum") -> None:
        self.datapath = datapath

    def get_data(self) -> DatasetDict:
        print(f"\033[92mLoading data from {self.datapath}\033[00m")

        data = load_dataset(self.datapath, trust_remote_code=True)
        return data
    

def ingest_data(datapath: str) -> DatasetDict:
    try:
        ingest_data = IngestDataset(datapath)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        print(f"Error while loading data: {e}")
        raise e
    

# if __name__=='__main__':
#     datapath = "knkarthick/dialogsum"
#     dataset = ingest_data(datapath)
#     print(dataset)
#     print(type(dataset))