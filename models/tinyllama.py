import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def load_llama_pipeline():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    return pipe, tokenizer
