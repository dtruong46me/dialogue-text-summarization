import logging
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# General class for BART and FLAN-T5
class GeneralModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)

    def generate(self, input_text, **kwargs):
        try:
            logger.info(f"Generating output...")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.base_model.generate(input_ids, **kwargs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Summary: {generated_text}")

            return generated_text

        except Exception as e:
            logger.error(f"Error while generating: {e}")
            raise e

    def prepare_for_int8(self):
        self.base_model = prepare_model_for_int8_training(self.base_model)

    def get_peft(self, lora_config):
        self.base_model = get_peft_model(self.base_model, lora_config)


# FLAN-T5 MODEL
class FlanT5Model(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)


# BART MODEL
class BartModel(GeneralModel):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)  



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
            logger.info(f"Load Bart model from checkpoint: {checkpoint}")
            return BartModel(checkpoint)
        
        if "flan" in checkpoint:
            logger.info(f"Load Flan-T5 model from checkpoint: {checkpoint}")
            return FlanT5Model(checkpoint)
        
        else:
            logger.info(f"Load general model from checkpoint: {checkpoint}")
            return GeneralModel(checkpoint)
        
    except Exception as e:
        logger.error("Error while loading model: {e}")
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