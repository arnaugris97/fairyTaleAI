import streamlit as st
from create_embedding_DB import MilvusEmbeddingProcessor
from inference_utils import load_model, preprocess_input, generate_embeddings, search_chromadb, generate_output
from langchain_community.llms import Ollama

model_path = 'Checkpoints/checkpoint.pt'
tokenizer_path = 'tokenizer/wordPieceVocab.json'

processor = MilvusEmbeddingProcessor(model_path, tokenizer_path, '')
llm = Ollama(model="llama3")

st.title("Fairy Tale Generator")

userPrompt = st.text_input("Enter a prompt to generate a fairy tale:", "Once upon a time there was a little red riding hood who lived in a small village.")
if st.button("Generate Fairy Tale"):
    with st.spinner('Generating fairy tale...'):
        try:
            results = processor.process_query(userPrompt)
            context = ''
            for result in results[0]:
                if context == '':
                    context += result['entity']['text']
                else:
                    context += ', ' + result['entity']['text']
            
        
            prompt_template = f"This is a prompt for a RAG system. I need you to create a fairy tale in catalan following the user prompt and the context. The prompt is the user prompt: {userPrompt} and this is the context: {context}. So the output should be in catalan."
            llm_result = llm.invoke(prompt_template)

            st.write("### Generated Fairy Tale")
            st.write(llm_result)
        except ConnectionError as e:
            st.error(f"Connection error: {e}")