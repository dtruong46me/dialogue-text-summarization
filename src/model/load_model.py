import logging

from bart import BartModel
from flant5 import FlanT5Model
from model import GeneralModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


if __name__=='__main__':
    checkpoint = "google/flan-t5-base"
    model = load_model(checkpoint)
    print(model)

    prompt = "Summarize the following conversation:\n\n#Person1#: Tell me something about your\
      Valentine's Day. #Person2#: Ok, on that day, boys usually give roses to the sweet hearts\
        and girls give them chocolate in return. #Person1#: So romantic. young people must have\
          lot of fun. #Person2#: Yeah, that is what the holiday is for, isn't it?\n\nSummary:"
    
    output1 = model.generate(prompt, min_new_tokens=120, max_length=256)
    output2 = model.generate(prompt, min_new_tokens=200, max_length=512)

    print(output1)
    print("\n\n")
    print(output2)