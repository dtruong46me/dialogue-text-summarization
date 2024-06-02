import streamlit as st
import replicate
import os
from transformers import AutoTokenizer, GenerationConfig, AutoModelForSeq2SeqLM
import torch

# Set Replicate API token
with st.sidebar:
    st.title('Dialogue Text Summarization')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your Replicate API token.', icon='⚠️')
            st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    st.subheader("Adjust model parameters")
    min_new_tokens = st.slider('Min new tokens', min_value=1, max_value=256, step=1, value=10)
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.00, step=0.01, value=1.0)
    top_k = st.slider('Top_k', min_value=1, max_value=50, step=1, value=20)
    top_p = st.slider('Top_p', min_value=0.01, max_value=1.00, step=0.01, value=1.0)

# Initialize model and tokenizer
checkpoint = "dtruong46me/train-bart-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

st.title("Dialogue Text Summarization")
st.caption("Natural Language Processing Project 20232")
st.write("---")

input_text = st.text_area("Dialogue", height=200)

generation_config = GenerationConfig(
    min_new_tokens=min_new_tokens,
    max_new_tokens=320,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k
)

def generate_summary(model, input_text, generation_config, tokenizer):
    prefix = "Summarize the following conversation: \n\n###"
    suffix = "\n\nSummary:"
    input_ids = tokenizer.encode(prefix + input_text + suffix, return_tensors="pt").to(model.device)
    prompt_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return prompt_str

def stream_summary(prompt_str, temperature, top_p):
    for event in replicate.stream(
        "snowflake/snowflake-arctic-instruct",
        input={"prompt": prompt_str,
               "prompt_template": r"{prompt}",
               "temperature": temperature,
               "top_p": top_p}):
        yield str(event['output'])

if st.button("Submit"):
    st.write("---")
    st.write("## Summary")

    if not replicate_api:
        st.error("Please enter your Replicate API token!")
    elif not input_text:
        st.error("Please enter a dialogue!")
    else:
        prompt_str = generate_summary(model, input_text, generation_config, tokenizer)
        summary_container = st.empty()

        summary_text = ""
        for output in stream_summary(prompt_str, temperature, top_p):
            summary_text += output
            summary_container.text(summary_text)
