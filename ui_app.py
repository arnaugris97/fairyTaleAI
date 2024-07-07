import streamlit as st
from API_secrets import ANTROPHIC_API_KEY, OPENAI_API_KEY
from createTL_embedding_DB import MilvusEmbeddingProcessorTL
from inference_utils import load_model, preprocess_input, generate_embeddings, search_chromadb, generate_output
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
import anthropic

model_path = 'Checkpoints/TL100Epochs.pt'
tokenizer_path = 'tokenizer/wordPieceVocab.json'

processor = MilvusEmbeddingProcessorTL(model_path, '')
llm = Ollama(model="llama3")

llmGPT = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-instruct", temperature=0)

client = anthropic.Anthropic(
    api_key=ANTROPHIC_API_KEY,
)

st.title("Fairy Tale Generator")
# Creating tabs
tab1, tab2 = st.tabs(["Fairy Tale Generator", "Detailed View"])

userPrompt = st.text_input("Enter a prompt to generate a fairy tale:", "Once upon a time there was a little red riding hood who lived in a small village.")

# Adding a toggle button to select between Llama 3 and ChatGPT
model_choice = st.radio("Select the inference model:", ("Llama 3", "GPT 3.5 Turbo", "Claude 3.5"))

if st.button("Generate Fairy Tale"):
    with st.spinner('Generating fairy tale...'):
        try:
            results, tokens = processor.process_query(userPrompt)
            context = ''
            for result in results[0]:
                
                    
                    if context == '':
                        context += result['entity']['text']
                    else:
                        context += ', ' + result['entity']['text']
            
        
            prompt_template = f"This is a prompt for a RAG system. I need you to create a fairy tale in catalan following the user prompt and the context. The prompt is the user prompt: {userPrompt} and this is the context: {context}. So the output should be in catalan."

            # Using the selected model for inference
            if model_choice == "Llama 3":
                llm_result = llm.invoke(prompt_template)
            elif model_choice == "ChatGPT":
                llm_result = llmGPT.invoke(prompt_template)
            else :
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt_template}
                    ]
                )
                llm_result = message.content[0].text
            

             # Tab 1: Fairy Tale Generator
            with tab1:
                st.write("### Generated Fairy Tale")
                st.write(llm_result)
            
            # Tab 2: Detailed View
            with tab2:
                st.write("### Tokens")
                st.write(tokens)
                st.write("### Context")
                st.write(context)
                st.write("### Prompt Sent to Llama 3")
                st.write(prompt_template)
                st.write("### Generated Fairy Tale")
                st.write(llm_result)
            
        except ConnectionError as e:
            st.error(f"Connection error: {e}")