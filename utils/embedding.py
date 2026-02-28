import numpy as np
from sentence_transformers import SentenceTransformer

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Turns text chunks into numerical vectors (embeddings).
    """
    # 1. Load the embedding model
    # 'all-MiniLM-L6-v2' is fast and runs locally
    model = SentenceTransformer(model_name)
    
    # 2. Embed all chunks
    # This converts text into a mathematical representation the computer understands
    chunk_embeddings = model.encode(chunks)
    
    print(f"Successfully created embeddings with shape: {chunk_embeddings.shape}")
    
    return chunk_embeddings, model

def get_query_embedding(model, query):
    """
    Converts a single user question into an embedding for searching.
    """
    # Convert query to float32 as required by FAISS
    return model.encode([query]).astype('float32')