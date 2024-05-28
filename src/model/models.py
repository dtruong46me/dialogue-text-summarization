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
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device)

# BART MODEL
class BartSumModel(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)  
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)


#FlanT5 model using LoRA
class FlanT5Model_LoRA(GeneralModel):
    def __init__(self, checkpoint, bnb_config):
        super().__init__(checkpoint)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, quantization_config= bnb_config, device_map={"":0}, trust_remote_code=True)

    def setup(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.checkpoint)
        self.base_model = BartModel.from_pretrained(self.checkpoint).to(self.device)

    def prepare_quantize(self):
        self.base_model = prepare_model_for_kbit_training(self.base_model)

    def get_peft(self, lora_config):
        self.base_model = get_peft_model(self.base_model, lora_config)


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

# if __name__=='__main__':
#     checkpoint = "google/flan-t5-base"
#     model = load_model(checkpoint)
#     print(model)

#     prompt = "Summarize the following conversation:\n\n#Person1#: Tell me something about your\
#       Valentine's Day. #Person2#: Ok, on that day, boys usually give roses to the sweet hearts\
#         and girls give them chocolate in return. #Person1#: So romantic. young people must have\
#           lot of fun. #Person2#: Yeah, that is what the holiday is for, isn't it?\n\nSummary:"
    
#     output1 = model.generate(prompt, min_new_tokens=120, max_length=256)
#     output2 = model.generate(prompt, min_new_tokens=200, max_length=512)

#     print(output1)
#     print("\n\n")
#     print(output2)