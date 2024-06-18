import streamlit as st
import pandas as pd

from transformers import GenerationConfig, BartModel, BartTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, TextStreamer
import torch
import time

import sys, os

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

st.title("Dialogue Text Summarization")
st.caption("Natural Language Processing Project 20232")

st.write("---") 


class StreamlitTextStreamer(TextStreamer):
    def __init__(self, tokenizer, st_container, st_info_container, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.st_container = st_container
        self.st_info_container = st_info_container
        self.text = ""
        self.start_time = None
        self.first_token_time = None
        self.total_tokens = 0

    def on_finalized_text(self, text: str, stream_end: bool=False):
        if self.start_time is None:
            self.start_time = time.time()

        if self.first_token_time is None and len(text.strip()) > 0:
            self.first_token_time = time.time()

        self.text += text

        self.total_tokens += len(text.split())
        self.st_container.markdown("###### " + self.text)
        time.sleep(0.03)

        if stream_end:
            total_time = time.time() - self.start_time
            first_token_wait_time = self.first_token_time - self.start_time if self.first_token_time else None
            tokens_per_second = self.total_tokens / total_time if total_time > 0 else None
            
            df = pd.DataFrame(data={
                "First token": [first_token_wait_time],
                "Total tokens": [self.total_tokens],
                "Time taken": [total_time],
                "Token per second": [tokens_per_second]
            })

            self.st_info_container.table(df)

def generate_summary(model, input_text, generation_config, tokenizer, st_container, st_info_container) -> str:
    try:
        prefix = "Summarize the following conversation: \n###\n"
        suffix = "\n### Summary:"
        target_length = max(1, int(0.15 * len(input_text.split())))

        input_ids = tokenizer.encode(prefix + input_text + f"The generated summary should be around {target_length} words." + suffix, return_tensors="pt")

        # Initialize the Streamlit container and streamer
        streamer = StreamlitTextStreamer(tokenizer, st_container, st_info_container, skip_special_tokens=True, decoder_start_token_id=3)

        model.generate(input_ids, streamer=streamer, do_sample=True, generation_config=generation_config)

    except Exception as e:
        raise e


with st.sidebar:
    checkpoint = st.selectbox("Model", options=[
        "Choose model",
        "dtruong46me/train-bart-base",
        "dtruong46me/flant5-small",
        "dtruong46me/flant5-base",
        "dtruong46me/flan-t5-s",
        "ntluongg/bart-base-luong"
    ])
    st.button("Model detail", use_container_width=True)
    st.write("-----")
    st.write("**Generate Options:**")
    min_new_tokens = st.number_input("Min new tokens", min_value=1, max_value=64, value=10)
    max_new_tokens = st.number_input("Max new tokens", min_value=64, max_value=128, value=64)
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    top_k = st.number_input("Top_k", min_value=1, max_value=50, step=1, value=20)
    top_p = st.number_input("Top_p", min_value=0.01, max_value=1.00, step=0.01, value=1.0)


height = 200

input_text = st.text_area("Dialogue", height=height)

generation_config = GenerationConfig(
    min_new_tokens=min_new_tokens,
    max_new_tokens=320,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if checkpoint=="Choose model":
    tokenizer = None
    model = None

if checkpoint!="Choose model":
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)



if st.button("Submit"):
    st.write("---")
    st.write("## Summary")

    if checkpoint=="Choose model":
        st.error("Please selece a model!")

    else:
        if input_text=="":
            st.error("Please enter a dialogue!")
        # generate_summary(model, " ".join(input_text.split()), generation_config, tokenizer)
        st_container = st.empty()
        st_info_container = st.empty()
        generate_summary(model, " ".join(input_text.split()), generation_config, tokenizer, st_container, st_info_container)