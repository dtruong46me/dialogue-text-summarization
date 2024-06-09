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
        "Bart Base",
        "Flan-T5 Small",
        "Flan-T5 Base",
        "Bart-QDS",
        "Flan-T5 QDS"
    ])
    st.button("Model detail", use_container_width=True)
    st.write("-----")
    st.write("**Generate Options:**")
    min_new_tokens = st.slider("Min new tokens", min_value=1, max_value=256, step=1, value=10)
    max_new_tokens = st.slider("Max new tokens", min_value=10, max_value=256, step=1, value=64)
    temperature = st.slider("Temperature", min_value=0.01, max_value=1.00, step=0.01, value=1.0)
    top_k = st.slider("Top_k", min_value=1, max_value=50, step=1, value=20)
    top_p = st.slider("Top_p", min_value=0.01, max_value=1.00, step=0.01, value=1.0)


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
    if checkpoint=="BART Base":
        checkpoint = "dtruong46me/train-bart-base"
    if checkpoint=="FLAN-T5 Small":
        checkpoint = "dtruong46me/flant5-small"
    if checkpoint=="FLAN-T5 Base":
        checkpoint = "dtruong46me/flant5-base"
    if checkpoint=="BART-QDS":
        checkpoint = "dtruong46me/bart-base-qds1"
    if checkpoint=="FLAN-T5 QDS":
        checkpoint = "dtruong46me/bart-base-qds"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)



if st.button("Submit"):
    st.write("---")
    st.write("## Summary")

    if checkpoint=="Choose model":
        st.error("Please selece a model!")

    else:
        if input_text=="":
            st.error("Please enter a dialogue!")
        st.write(generate_summary(model, " ".join(input_text.split()), generation_config, tokenizer))