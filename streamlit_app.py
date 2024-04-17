import streamlit as st

st.title("Dialogue Text Summarization")
st.caption("Natural Language Processing Project 20232")

st.write("---")

with st.sidebar:
    st.selectbox("Model", options=[
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "facebook/bart-base"
    ])
    st.button("Model detail", use_container_width=True)
    st.write("-----")
    st.write("**Generate Options:**")
    st.slider("Temperature", min_value=0.01, max_value=1.00, step=0.01)
    st.slider("Top_k", min_value=1, max_value=20)
    st.slider("Top_p", min_value=0.01, max_value=1.00, step=0.01)

dialogue = "#Person1#: But you should tell me you were in love with her. #Person2#: Didn't I? #Person1#: You know you didn't. #Person2#: Well, I am telling you now. #Person1#: Yes, but you might have told me before. #Person2#: I didn't think you would be interested. #Person1#: You can't be serious. How dare you not tell me you are going to marry her? #Person2#: Sorry, I didn't think it mattered. #Person1#: Oh, you men! You are all the same."

dialogue_with_newlines = '#Person' + '\n#Person'.join([line for line in dialogue.split('#Person') if line.strip()])
num_lines = dialogue_with_newlines.count('\n') + 1
height = min(num_lines * 26, 360)

input = st.text_area("Dialogue", value=dialogue_with_newlines, height=height)
print(input)

col1, col2 = st.columns([1,6])
with col1:
    st.button("Submit")

with col2:
    st.button("Clear")

st.success("#Person1#'s angry because #Person2# didn't tell #Person1# that #Person2# had a girlfriend and would marry her. ")