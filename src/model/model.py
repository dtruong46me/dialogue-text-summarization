import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BartModel
)

from peft import (
    get_peft_model,
    prepare_model_for_kbit_training
)

class Model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self):
        pass

    def get_peft(self, lora_config):
        self.base_model = get_peft_model(self.base_model, lora_config)
    
    def prepare_quantize(self, bnb_config):
        if "bart" in self.checkpoint:
            self.base_model = BartModel.from_pretrained(self.checkpoint, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
        if "flan" in self.checkpoint:
            self.base_model =  AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint, quantization_config= bnb_config, device_map={"":0}, trust_remote_code=True)

        self.base_model = prepare_model_for_kbit_training(self.base_model)


class BartSum(Model):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)

        self.tokenizer = BartTokenizer.from_pretrained(self.checkpoint)

    def get_model(self):
        self.base_model = BartModel.from_pretrained(self.checkpoint).to(self.device)


class FlanT5Sum(Model):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def get_model(self):
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device)


def load_model(checkpoint):

    try:
        if "bart" in checkpoint:
            print(f"\033[92mLoad Bart model from checkpoint: {checkpoint}\033[00m")
            return BartSum(checkpoint)
        
        if "flan" in checkpoint:
            print(f"\033[92mLoad Flan-T5 model from checkpoint: {checkpoint}\033[00m")
            return FlanT5Sum(checkpoint)
        
        else:
            print(f"\033[92mLoad general model from checkpoint: {checkpoint}\033[00m")
            return Model(checkpoint)
        
    except Exception as e:
        print("Error while loading model: {e}")
        raise e