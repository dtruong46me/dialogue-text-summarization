import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from peft import (
    get_peft_model,
)

class Model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.base_model = None

    def get_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)

    def get_peft(self, lora_config):
        return get_peft_model(self.base_model, lora_config)
    
    def prepare_quantize(self, bnb_config):
        return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint, 
                                                                 quantization_config=bnb_config, 
                                                                 device_map={"":0}, 
                                                                 trust_remote_code=True)
        # self.base_model.gradient_checkpointing_enable()
        # self.base_model = prepare_model_for_kbit_training(self.base_model)


    def generate_summary(self, input_text, generation_config, do_sample=True):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
        output_ids = self.base_model.generate(input_ids=input_ids, do_sample=do_sample, generation_config=generation_config)
        
        if "bart" in self.checkpoint:
            output_ids[0][1] = 2

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\033[94mSummary: {output_text}\n\033[00m")
        return output_text

class BartSum(Model):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def get_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)


class FlanT5Sum(Model):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def get_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)


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