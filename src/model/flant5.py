import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

logger = logging.getLogger(__name__)


class FineTuned_t5Model:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.t5models = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)

    def generate(self, input_text, **kwargs):
        try:
            logger.info("Generating output for input: {input_text}")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.t5models.generate(input_ids, **kwargs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logger.error(f"Error while generating: {e}")
            raise e


def load_model(model_id):
    try:
        return FineTuned_t5Model(model_id)
    except Exception as e:
        logger.error("Error while loading model: {e}")
        raise e


if __name__ == '__main__':
    model_id = "google/flan-t5-base"
    model = load_model(model_id)

    #     # prompt = "def print_hello_world():"
    #     # prompt = "Write a function to get a lucid number smaller than or equal to n."
    prompt = "summarize: #Person1#: Why didn't you tell me you had a girlfriend? #Person2#: Sorry, I thought you " \
             "knew. #Person1#: But you should tell me you were in love with her. #Person2#: Didn't I? #Person1#: You " \
             "know you didn't. #Person2#: Well, I am telling you now. #Person1#: Yes, but you might have told me " \
             "before. #Person2#: I didn't think you would be interested. #Person1#: You can't be serious. How dare " \
             "you not tell me you are going to marry her? #Person2#: Sorry, I didn't think it mattered. #Person1#: " \
             "Oh, you men! You are all the same."

    output = model.generate(prompt, temperature=0.7, min_new_tokens=64, top_k=2)
    print(output)
#     print()
#     print(type(model))
