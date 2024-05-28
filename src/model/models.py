import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import BartTokenizer, BartModel

from peft import get_peft_model, prepare_model_for_kbit_training


# General class for BART and FLAN-T5
class GeneralModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None 
        self.base_model = None

    def setup(self):
        pass

    def get_peft(self, lora_config):
        self.base_model = get_peft_model(self.base_model, lora_config)

    def prepare_quantize(self, bnb_config):
        if "bart" in self.checkpoint:
            self.base_model = BartModel.from_pretrained(self.checkpoint, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
        if "flan" in self.checkpoint:
            self.base_model =  AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint, quantization_config= bnb_config, device_map={"":0}, trust_remote_code=True)
        
        self.base_model = prepare_model_for_kbit_training(self.base_model)


# FLAN-T5 MODEL
class FlanT5SumModel(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def setup(self):
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device)

# BART MODEL
class BartSumModel(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        print("self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def setup(self):
        print("self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device)")
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device)
        self.base_model.generation_config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.base_model.generation_config.bos_token_id = self.tokenizer.cls_token_id

        print(self.base_model.generation_config)


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