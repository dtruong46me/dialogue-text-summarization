
from datasets import load_dataset
from datasets import DatasetDict, Dataset
import random
from transformers import set_seed


def ingest_data(datapath: str) -> DatasetDict:
    set_seed(42)

    QDS_LIMIT = 6000
    if "," in datapath:
        datapaths = datapath.split(",")
    
    datapath1 = "binwang/InstructDS_datasets"
    datapath2 = "binwang/InstructDS_datasets"

    all_train_data = []
    origin_train_dialogsum = load_dataset(datapath1, "DialogSum", split="train")
    qds_dialogsum = load_dataset(datapath2, "DialogSum_QDS", split="train")

    new_data1 = []
    for sample in origin_train_dialogsum:
        new_sample = {
            "instruction": "Please summarize the following dialogue.",
            "input": sample["dialogue"],
            "output": sample["summary"]
        }
        new_data1.append(new_sample)
    origin_train_dialogsum = new_data1
    all_train_data.extend(origin_train_dialogsum)
    
    new_data2 = []
    for sample in qds_dialogsum:
        new_sample = {
            "instruction": "Please answer the following question.",
            "input": sample["dialogue"],
            "output": sample["summary"]
        }
        new_data2.append(new_sample)
    qds_dialogsum = new_data2
    all_train_data.extend(qds_dialogsum)


    naive_all_train_data_dict = {
        "instruction": [item["instruction"] for item in all_train_data],
        "input": [item["input"] for item in all_train_data],
        "output": [item["output"] for item in all_train_data]
    }

    subset_train_data = random.sample(all_train_data, QDS_LIMIT)
    with_len_train_data_dict = {
        "instruction": [item["instruction"] + f" The output should be {len(item['output'].split())} words long." for item in subset_train_data],
        "input": [item["input"] for item in subset_train_data],
        "output": [item["output"] for item in subset_train_data]
    }

    all_train_data_dict = {
        "instruction": naive_all_train_data_dict["instruction"] + with_len_train_data_dict["instruction"],
        "input": naive_all_train_data_dict["input"] + with_len_train_data_dict["input"],
        "output": naive_all_train_data_dict["output"] + with_len_train_data_dict["output"]
    }

    raw_train_data = Dataset.from_dict(all_train_data_dict)
    train_data = raw_train_data.shuffle()


    # Validation data
    all_validation_data = []
    origin_validation_dialogsum = load_dataset(datapath1, "DialogSum", split="validation")

    new_data1 = []
    for sample in origin_validation_dialogsum:
        new_sample = {
            "instruction": "Please summarize the following dialogue.",
            "input": sample["dialogue"],
            "output": sample["summary"]
        }
        new_data1.append(new_sample)
    
    origin_validation_dialogsum = new_data1
    all_validation_data.extend(origin_validation_dialogsum)

    all_validation_data_dict = {
        "instruction": [item["instruction"] for item in all_validation_data],
        "input": [item["input"] for item in all_validation_data],
        "output": [item["output"] for item in all_validation_data]
    }

    raw_validation_data = Dataset.from_dict(all_validation_data_dict)
    validation_data = raw_validation_data.shuffle()

    return DatasetDict({
        "train": train_data,
        "validation": validation_data
    })