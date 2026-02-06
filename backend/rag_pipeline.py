
import os
import re
import pickle
from typing import List, Tuple

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ===========================================
# CONFIG
# ===========================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_CONTEXT_CHARS = 3500

PDF_FOLDER = "data/pdfs"  # folder for your synthetic PDFs
EMBEDDINGS_FILE = "data/embeddings.pkl"
FAISS_INDEX_FILE = "data/faiss.index"

HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face API token
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # small hosted model

client = InferenceClient(token=HF_TOKEN)


# ===========================================
# PDF & TEXT CHUNKING
# ===========================================
def load_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = ""

    for sent in sentences:
        if len(chunk) + len(sent) > chunk_size:
            chunks.append(chunk.strip())
            chunk = chunk[-overlap:]
        chunk += " " + sent

    if chunk.strip():
        chunks.append(chunk.strip())

    return chunks


def load_pdfs(folder_path: str) -> Tuple[List[str], List[str]]:
    """Load all PDFs in a folder and return chunks + source filenames."""
    chunks, sources = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            text = load_pdf_text(full_path)
            pdf_chunks = chunk_text(text)
            chunks.extend(pdf_chunks)
            sources.extend([filename] * len(pdf_chunks))
    return chunks, sources


# ===========================================
# EMBEDDINGS & FAISS
# ===========================================
def save_embeddings(data, path=EMBEDDINGS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_embeddings(path=EMBEDDINGS_FILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_faiss(index, path=FAISS_INDEX_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss(path=FAISS_INDEX_FILE):
    if os.path.exists(path):
        return faiss.read_index(path)
    return None


def build_resources():
    """Load or build embeddings and FAISS index."""
    embedder = SentenceTransformer("all-mpnet-base-v2")

    cached = load_embeddings()
    if cached:
        chunks, sources, embeddings = cached
    else:
        chunks, sources = load_pdfs(PDF_FOLDER)
        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        save_embeddings((chunks, sources, embeddings))

    index = load_faiss()
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        save_faiss(index)

    return chunks, sources, embedder, index


def search(query: str, embedder, index, chunks, sources, k=TOP_K) -> List[Tuple[str, str]]:
    """Search top-k chunks from FAISS index based on query."""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    _, idxs = index.search(q_emb, k)
    return [(chunks[i], sources[i]) for i in idxs[0]]


# ===========================================
# HUGGING FACE LLM QUERY
# ===========================================
def query_llm(prompt: str) -> str:
    """Call a hosted Hugging Face model and return text."""
    try:
        response = client.text_generation(
            model=HF_MODEL,
            prompt=prompt,
            max_new_tokens=500
        )
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error querying LLM: {e}"


# ===========================================
# RAG PIPELINE
# ===========================================
def answer_question(question: str, resources=None) -> Tuple[str, List[str]]:
    """
    Given a user question, return answer and sources.
    resources: tuple(chunks, sources, embedder, index)
    """
    if resources is None:
        resources = build_resources()
    chunks, sources, embedder, index = resources

    # Search top chunks
    results = search(question, embedder, index, chunks, sources)

    # Build context
    context = ""
    used_sources = set()
    for chunk, src in results:
        if len(context) + len(chunk) > MAX_CONTEXT_CHARS:
            break
        context += f"\n[{src}]\n{chunk}\n"
        used_sources.add(src)

    # Build prompt
    prompt = f"""
You are a helpful AI assistant answering strictly from the context below.

If the answer is NOT explicitly stated, say:
"I don’t know based on the documents provided."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # Query LLM
    answer = query_llm(prompt)

    # Clean answer
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    answer = re.sub(r"^Answer:\s*", "", answer, flags=re.I).strip()

    return answer, list(used_sources)
