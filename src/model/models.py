import logging
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import BartTokenizer, BartModel

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# General class for BART and FLAN-T5
class GeneralModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None 
        self.base_model = None

    def setup(self):
        pass

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.generate(input_ids, attention_mask=attention_mask)
        return outputs

    def get_peft(self, lora_config):
        self.base_model = get_peft_model(self.base_model, lora_config)

    def prepare_quantize(self):
        self.base_model = prepare_model_for_kbit_training(self.base_model)


    # def generate_summary(self, input_text, **kwargs):
    #     try:
    #         print(f"\033[92mGenerating output...\033[00m")
    #         input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
    #         outputs = self.base_model.generate(input_ids, do_sample=True, **kwargs)
    #         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         print(f"\033[92mSummary: {generated_text}\033[00m")

    #         return generated_text

    #     except Exception as e:
    #         print(f"Error while generating: {e}")
    #         raise e


# FLAN-T5 MODEL
class FlanT5SumModel(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device)

# BART MODEL
class BartSumModel(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)  

    def setup(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.checkpoint)
        self.base_model = BartModel.from_pretrained(self.checkpoint).to(self.device)


def load_model(checkpoint):
    """
    Loads a model base on the `checkpoint` and optionally the `model_type`

    Args:
        checkpoint (str): the checkpoint from huggingface
        model_type (str, optional): Specific the model type (e.g. "bart" or "flan-t5")
    
    Returns:
        GeneralModel: The loaded model instance
    """
    try:
        if "bart" in checkpoint:
            print(f"\033[92mLoad Bart model from checkpoint: {checkpoint}\033[00m")
            return BartSumModel(checkpoint)
        
        if "flan" in checkpoint:
            print(f"\033[92mLoad Flan-T5 model from checkpoint: {checkpoint}\033[00m")
            return FlanT5SumModel(checkpoint)
        
        else:
            print(f"\033[92mLoad general model from checkpoint: {checkpoint}\033[00m")
            return GeneralModel(checkpoint)
        
    except Exception as e:
        print("Error while loading model: {e}")
        raise e