import logging
from datasets import load_dataset
from datasets import DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestDataset:
    def __init__(self, datapath: str="knkarthick/dialogsum") -> None:
        self.datapath = datapath

    def get_data(self) -> DatasetDict:
        logger.info(f"Loading data from {self.datapath}")

        data = load_dataset(self.datapath, trust_remote_code=True)
        logger.info(f"Complete loading data from {self.datapath}")
        
        return data
    

def ingest_data(datapath: str) -> DatasetDict:
    try:
        ingest_data = IngestDataset(datapath)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise e
    

# if __name__=='__main__':
#     datapath = "knkarthick/dialogsum"
#     dataset = ingest_data(datapath)
#     print(dataset)
#     print(type(dataset))