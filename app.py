import streamlit as st
from loaders.pdf_loader import load_and_chunk_pdfs
from vectorstore.faiss_store import create_vector_store
from models.tinyllama import load_llama_pipeline
from utils.prompt_builder import build_prompt

@st.cache_resource(show_spinner="hold on i'm learning")
def setup():
    chunks = load_and_chunk_pdfs()
    vectorstore = create_vector_store(chunks)
    pipe, tokenizer = load_llama_pipeline()
    return vectorstore, pipe, tokenizer

st.title("kairos")
st.caption("/kʌɪrɒs/ the right moment")

query = st.text_input("ask me: ")

if query:
    with st.spinner("i'm thinking..."):
        vectorstore, pipe, tokenizer = setup()
        relevant_docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = build_prompt(context, query, tokenizer)

        output = pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_k=10,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated = output[0]["generated_text"].replace(prompt, "").strip()

    st.success(generated)
