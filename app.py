import streamlit as st

from transformers import GenerationConfig, BartModel, BartTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

import sys, os

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from gen_summary import generate_summary


st.title("Dialogue Text Summarization")
st.caption("Natural Language Processing Project 20232")

st.write("---")

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
        st.write(generate_summary(model, " ".join(input_text.split()), generation_config, tokenizer))