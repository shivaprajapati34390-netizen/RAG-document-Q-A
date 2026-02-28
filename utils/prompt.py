def create_rag_prompt(query, context):
    """
    Combines retrieved context with the user query into a structured prompt.
    """
    # This template ensures the AI stays grounded in your data
    # It explicitly instructs the model to avoid guessing if the info is missing
    prompt_template = f"""
    answer the following question using only the provided context.
    if the answer is not in the context,say "i don't have any information"

    Context:
    {context}

    Question:
    {query}
    Answer:
    """
    
    return prompt_template

def format_context(retrieved_chunks):
    """
    Joins multiple text chunks into a single string for the prompt.
    """
    # Uses a newline and period to separate chunks clearly for the model
    return "\n\n.".join(retrieved_chunks)