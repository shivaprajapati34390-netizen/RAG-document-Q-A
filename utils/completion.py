from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_generator_model(model_name='google/flan-t5-small'):
    """
    Loads the tokenizer and the LLM for generating answers.
    """
    # Load the tokenizer and model for flan-t5-small
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def get_answer(prompt, tokenizer, model):
    """
    Takes the prompt and uses the model to generate the final response.
    """
    # 1. Encode the prompt into a format the model understands
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    # 2. Generate the output sequence
    # max_new_tokens=100 ensures the answer isn't cut off
    # do_sample=False keeps the output deterministic (consistent)
    outputs = model.generate(input_ids, max_new_tokens=100, do_sample=False)
    
    # 3. Convert the generated IDs back into plain text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer