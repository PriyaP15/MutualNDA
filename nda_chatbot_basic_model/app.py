import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import Together
import os

#  Set your Together AI API key
os.environ["TOGETHER_API_KEY"] = st.secrets["together_api_key"]

#  Initialize the Together AI model
llm = Together(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    max_tokens=100
)

st.set_page_config(page_title="Mutual NDA Chatbot", layout="centered")
st.title("🤖 Mutual NDA Chatbot")
st.write("Upload a Mutual NDA PDF and ask questions about it!")

#  Upload PDF
pdf = st.file_uploader("Upload your NDA PDF", type="pdf")

if pdf:
    # Read text from PDF
    pdf_reader = PdfReader(pdf)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    st.success("✅ PDF uploaded and read successfully!")

    #questions
    question = st.text_input("Ask a question about the NDA:")

    if question:
        prompt = f"""You are a legal assistant. Based on the following NDA document, answer the user's question.

        NDA Document:
        {pdf_text}

        User Question:
        {question}

        Answer:"""

        try:
            answer = llm.invoke(prompt)
            st.markdown("### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f" Error: {str(e)}")
