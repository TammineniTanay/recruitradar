# rag.py
# This file does 4 things:
# 1. Reads the resume PDF
# 2. Splits it into small chunks
# 3. Converts chunks to vectors and stores in Qdrant
# 4. Retrieves relevant chunks when given a job description

import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load the API key from .env file
load_dotenv()

# Constants - these are settings we'll use throughout
COLLECTION_NAME = "resume_chunks"  # Name of our storage bucket in Qdrant
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small fast model that converts text to vectors
RESUME_PATH = "resume.pdf"

def load_resume():
    """
    Reads the PDF and extracts all text from every page.
    Think of this as opening the PDF and copying all the text out.
    """
    reader = PdfReader(RESUME_PATH)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text

def split_into_chunks(text):
    """
    Splits the full resume text into small overlapping pieces.
    
    Why overlap? So we don't lose context at the boundaries.
    Example: chunk 1 = lines 1-10, chunk 2 = lines 8-18 (overlap at 8-10)
    
    chunk_size=500 = each chunk is ~500 characters
    chunk_overlap=100 = chunks share 100 characters with neighbors
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.create_documents([text])
    print(f"✅ Split resume into {len(chunks)} chunks")
    return chunks

def get_embeddings():
    """
    Loads the embedding model.
    This model converts text into vectors (lists of numbers).
    Example: "Python developer" → [0.23, -0.45, 0.87, ...]
    Similar texts get similar vectors - that's how semantic search works.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vector_store(chunks, embeddings):
    """
    Creates a Qdrant vector store in memory and stores all chunks.
    
    Qdrant is our vector database - it stores all the chunks as vectors
    and can quickly find the most similar ones to any query.
    
    We use :memory: so it runs locally without needing a server.
    """
    client = QdrantClient(":memory:")
    
    # Create a collection (like a table in a regular database)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # all-MiniLM-L6-v2 produces 384-dimensional vectors
            distance=Distance.COSINE  # Cosine similarity = best for text comparison
        )
    )
    
    # Store all chunks as vectors in Qdrant
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_documents(chunks)
    print(f"✅ Stored {len(chunks)} chunks in Qdrant vector store")
    return vector_store

def retrieve_relevant_chunks(vector_store, job_description, k=5):
    """
    Given a job description, finds the k most relevant resume chunks.
    
    How it works:
    1. Converts the job description to a vector
    2. Compares it against all resume chunk vectors
    3. Returns the top k most similar chunks
    
    k=5 means we get the 5 most relevant resume sections
    """
    results = vector_store.similarity_search(job_description, k=k)
    print(f"✅ Retrieved {len(results)} relevant chunks")
    return results

def initialize_rag():
    """
    Master function that runs the full pipeline:
    Load PDF → Split → Embed → Store → Return ready vector store
    Call this once when the app starts.
    """
    print("🚀 Initializing RAG pipeline...")
    text = load_resume()
    print(f"✅ Loaded resume: {len(text)} characters")
    chunks = split_into_chunks(text)
    embeddings = get_embeddings()
    print("✅ Embedding model loaded")
    vector_store = build_vector_store(chunks, embeddings)
    print("🎯 RAG pipeline ready!")
    return vector_store