import streamlit as st
import pdfplumber
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------
# CONFIG
# ----------------------------
import os

@st.cache_resource
def load_groq():
    return Groq(api_key=st.secrets["GROK"])

client = load_groq()

# ----------------------------
# LOAD MODELS (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_groq():
    return Groq(api_key=GROQ_API_KEY)

model = load_model()
client = load_groq()

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

# ----------------------------
# CHUNKING
# ----------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ----------------------------
# CREATE FAISS INDEX
# ----------------------------
def create_index(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, embeddings

# ----------------------------
# RETRIEVE
# ----------------------------
def retrieve(query, index, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# ----------------------------
# GROQ RESPONSE
# ----------------------------
def ask_groq(context, question):
    prompt = f"""
You are a friendly AI Tutor.

1. Explain in simple words
2. Give an example
3. Ask a short question to test understanding

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📚 RAG AI Tutor (PDF Based)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        index, _ = create_index(chunks)

    st.success("PDF processed successfully!")

    query = st.text_input("Ask a question from your PDF:")

    if query:
        with st.spinner("Thinking..."):
            retrieved_chunks = retrieve(query, index, chunks)
            context = "\n".join(retrieved_chunks)
            answer = ask_groq(context, query)

        st.write("### 📖 Answer")
        st.write(answer)
