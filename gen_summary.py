import torch
from transformers import AutoTokenizer, GenerationConfig, TextStreamer, AutoModelForSeq2SeqLM

import logging

import warnings
warnings.filterwarnings("ignore")

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def generate_summary(model, input_text, generation_config, tokenizer, st_container=None) -> str:

    try:
        prefix = "Summarize the following conversation: \n###\n"
        suffix = "\n### Summary:"

        input_ids = tokenizer.encode(prefix + input_text + "The generated summary should be around " + str(0.15*len(input_text)) + " words." + suffix, return_tensors="pt")
        output_ids = model.generate(input_ids, do_sample=True, generation_config=generation_config)

        if "bart" in model.name_or_path:
            output_ids[0][1] = 2
        
        # streamer = TextStreamer(tokenizer, skip_special_tokens=True)
        # model.generate(input_ids, streamer=streamer, do_sample=True, decoder_start_token_id=2, generation_config=generation_config)
        # logger.info("\nComplete generate summary!")
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    
    except Exception as e:
        print(f"Error while generating: {e}")
        raise e
    
if __name__=="__main__":
    input = "#Person1#: Ms. Dawson, I need you to take a dictation for me. #Person2#: Yes, sir... #Person1#: This should go out as an intra-office memorandum to all employees by this afternoon. Are you ready? #Person2#: Yes, sir. Go ahead. #Person1#: Attention all staff... Effective immediately, all office communications are restricted to email correspondence and official memos. The use of Instant Message programs by employees during working hours is strictly prohibited. #Person2#: Sir, does this apply to intra-office communications only? Or will it also restrict external communications? #Person1#: It should apply to all communications, not only in this office between employees, but also any outside communications. #Person2#: But sir, many employees use Instant Messaging to communicate with their clients. #Person1#: They will just have to change their communication methods. I don't want any - one using Instant Messaging in this office. It wastes too much time! Now, please continue with the memo. Where were we? #Person2#: This applies to internal and external communications. #Person1#: Yes. Any employee who persists in using Instant Messaging will first receive a warning and be placed on probation. At second offense, the employee will face termination. Any questions regarding this new policy may be directed to department heads. #Person2#: Is that all? #Person1#: Yes. Please get this memo typed up and distributed to all employees before 4 pm."
    target1 = "Ms. Dawson helps #Person1# to write a memo to inform every employee that they have to change the communication method and should not use Instant Messaging anymore."
    target2 = "In order to prevent employees from wasting time on Instant Message programs, #Person1# decides to terminate the use of those programs and asks Ms. Dawson to send out a memo to all employees by the afternoon."
    target3 = "Ms. Dawson takes a dictation for #Person1# about prohibiting the use of Instant Message programs in the office. They argue about its reasonability but #Person1# still insists."

    generation_config = GenerationConfig(
        min_new_tokens=10,
        max_new_tokens=256,
        temperature=0.9,
        top_p=1.0,
        top_k=50        
    )

    checkpoint = "dtruong46me/bart-base-qds2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    
    generate_summary(model, input, generation_config, tokenizer)
    print("\n==============\n")

    print("Human base line:\n", target1, end="\n\n")
    print("Human base line:\n", target2, end="\n\n")
    print("Human base line:\n", target3, end="\n\n")