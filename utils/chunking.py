from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_chunks(text, chunk_size=150, chunk_overlap=20):
    """
    Splits a long string of text into smaller chunks.
    """
    # 1. Initialize the Text Splitter
    # This splitter looks for natural breaks like double newlines or spaces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,    # Max size of a chunk
        chunk_overlap=chunk_overlap, # Overlap to maintain context between chunks
        length_function=len       # Uses character count
    )
    
    # 2. Create the chunks
    chunks = text_splitter.split_text(text)
    
    print(f"Original text split into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test script for local debugging
    test_text = "Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data."
    test_chunks = get_text_chunks(test_text)
    for i, c in enumerate(test_chunks):
        print(f"Chunk {i+1}: {c}")