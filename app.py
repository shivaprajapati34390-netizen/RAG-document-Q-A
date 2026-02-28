import streamlit as st
import os
from utils.chunking import get_text_chunks
from utils.embedding import create_embeddings, get_query_embedding
from utils.retrieval import create_vector_store, retrieve_relevant_chunks
from utils.prompt import create_rag_prompt, format_context
from utils.completion import load_generator_model, get_answer

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Document Chatbot", page_icon="📄")
st.title("📄 RAG Document Q&A")
st.markdown("Upload a text file to chat with your data using **Flan-T5** and **FAISS**.")

# --- 1. LOAD GENERATOR MODEL (Cached) ---
@st.cache_resource
def setup_models():
    # Loading the Flan-T5 model and tokenizer
    tokenizer, model = load_generator_model()
    return tokenizer, model

tokenizer, model_generator = setup_models()

# --- 2. FILE UPLOAD & PROCESSING ---
uploaded_file = st.file_uploader("Upload your .txt file", type=['txt'])

if uploaded_file:
    # Read and decode the file content
    file_content = uploaded_file.read().decode("utf-8")
    
    # Run our modular pipeline
    # A. Chunking
    chunks = get_text_chunks(file_content)
    
    # B. Embedding
    embeddings, embed_model = create_embeddings(chunks)
    
    # C. Vector Store
    index = create_vector_store(embeddings)
    
    st.success(f"Processing complete! Created {len(chunks)} searchable chunks.")

    # --- 3. CHAT INTERFACE ---
    user_query = st.text_input("Ask a question about your document:")
    
    if user_query:
        with st.spinner("Thinking..."):
            # D. Retrieval
            query_vec = get_query_embedding(embed_model, user_query)
            relevant_chunks = retrieve_relevant_chunks(query_vec, index, chunks, k=2)
            
            # E. Prompt Engineering
            context = format_context(relevant_chunks)
            final_prompt = create_rag_prompt(user_query, context)
            
            # F. Generation
            answer = get_answer(final_prompt, tokenizer, model_generator)
            
            # Display Results
            st.write("### Answer:")
            st.info(answer)
            
            # Optional: Show the context used
            with st.expander("View Retrieved Context"):
                st.write(context)