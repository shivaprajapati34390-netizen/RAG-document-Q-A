import faiss
import numpy as np

def create_vector_store(chunk_embeddings):
    """
    Initializes a FAISS index and adds the chunk embeddings to it.
    """
    # 1. Get the dimension of the embeddings (e.g., 384 for all-MiniLM-L6-v2)
    d = chunk_embeddings.shape[1]
    
    # 2. Create the FAISS index
    # IndexFlatL2 calculates the exact L2 (Euclidean) distance
    index = faiss.IndexFlatL2(d)
    
    # 3. Add chunk embeddings to the index
    # Note: Vectors must be converted to float32 for FAISS
    index.add(np.array(chunk_embeddings).astype(np.float32))
    
    print(f"FAISS index created with {index.ntotal} vectors.")
    return index

def retrieve_relevant_chunks(query_embedding, index, chunks, k=2):
    """
    Searches the FAISS index for the 'k' most similar chunks to the query.
    """
    # 1. Perform the search
    # 'distances' contains similarity scores, 'indices' contains the position of the chunks
    distances, indices = index.search(query_embedding, k)
    
    # 2. Get the actual text from the original chunks list using the found indices
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    return retrieved_chunks