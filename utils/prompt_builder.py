def build_prompt(context, question, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with expertise in climate science."},
        {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
