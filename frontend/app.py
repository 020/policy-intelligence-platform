import streamlit as st
from huggingface_hub import InferenceClient

HF_TOKEN = "YOUR_HF_TOKEN"
client = InferenceClient(token=HF_TOKEN)

st.title("🤖 Policy Chatbot (Demo)")

question = st.text_input("Ask a policy question:")

if question:
    with st.spinner("Thinking..."):
        response = client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            prompt=question,
            max_new_tokens=200
        )
    st.markdown("### Answer")
    st.write(response[0]["generated_text"])
