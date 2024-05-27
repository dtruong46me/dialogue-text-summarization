import torch
from transformers import AutoTokenizer

import os, sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from model.models import GeneralModel, FlanT5Model, BartModel

def generate_summary(model, input_text, generation_config, tokenizer, device):
    """
    Generate a summary given an input text
    """
    try:
        print(input_text)
        print(f"\033[92mGenerating output...\033[00m")
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        outputs = model.generate(input_ids, do_sample=True, **generation_config)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\033[92mSummary: {generated_text}\033[00m")
        return generated_text
    
    except Exception as e:
        print(f"Error while generating: {e}")
        raise e
    
if __name__=="__main__":
    input1 = ""
    input2 = ""
    input3 = ""
    input4 = ""

    model = BartModel(checkpoint="facebook/bart-base")
    generation_config = {
        "length": 50,
        "num_beams": 4,
        "early_stopping": True,
        "num_return_sequences": 1,
        "max_length": 50
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base").to(device)
    
    output1 = generate_summary(model, input1, generation_config, tokenizer, device)
    output2 = generate_summary(model, input2, generation_config, tokenizer, device)
    output3 = generate_summary(model, input3, generation_config, tokenizer, device)
    output4 = generate_summary(model, input4, generation_config, tokenizer, device)

    print(output1)
    print(output2)
    print(output3)
    print(output4)