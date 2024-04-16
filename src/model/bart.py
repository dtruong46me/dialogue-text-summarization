import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

logger = logging.getLogger(__name__)


class FineTuned_BART():
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bartmodels = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)

    def generate(self, input_text, **kwargs):
        try:
            logger.info("Generating output for input: {input_text}")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.bartmodels.generate(input_ids, **kwargs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logger.error(f"Error while generating: {e}")
            raise e


def load_model(model_id):
    try:
        return FineTuned_BART(model_id)
    except Exception as e:
        logger.error("Error while loading model: {e}")
        raise e
