import torch
from transformers import AutoTokenizer, GenerationConfig, BartModel


def generate_summary(model, input_text, generation_config, tokenizer) -> str:

    try:
        prefix = "Summarize the following conversation: \n\n###"
        suffix = "\n\nSummary:"
        tokenized_text = model.generate(
            tokenizer.encode(prefix + input_text + suffix, return_tensors="pt"),
            do_sample=True,
            generation_config=generation_config
        )

        generated_text = tokenizer.decode(tokenized_text[0], skip_special_tokens=True)
        
        return generated_text
    
    except Exception as e:
        print(f"Error while generating: {e}")
        raise e
    
if __name__=="__main__":
    input1 = ""
    input2 = ""
    input3 = ""
    input4 = ""

    generation_config = GenerationConfig(
        min_new_tokens=10,
        max_new_tokens=256,
        temperature=0.9,
        top_p=1.0,
        top_k=50
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base").to(device)

    model = BartModel.from_pretrained("facebook/bart-base").to(device)
    
    output1 = generate_summary(model, input1, generation_config, tokenizer)
    output2 = generate_summary(model, input2, generation_config, tokenizer)
    output3 = generate_summary(model, input3, generation_config, tokenizer)
    output4 = generate_summary(model, input4, generation_config, tokenizer)

    print(output1)
    print(output2)
    print(output3)
    print(output4)