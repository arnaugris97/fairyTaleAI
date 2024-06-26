from create_embedding_DB import MilvusEmbeddingProcessor
from inference_utils import load_model, preprocess_input, generate_embeddings, search_chromadb, generate_output
import chromadb
from langchain_community.llms import Ollama


if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'

    processor = MilvusEmbeddingProcessor(model_path, tokenizer_path, '')
    llm = Ollama(model="llama3")

    userPrompt = "Once upon a time there was a little riding hood who lived in a small village."
    
    try:
        results = processor.process_query(userPrompt)
        context = ''
        for result in results[0]:
            if context == '':
                context += result['entity']['text']
            else:
                context += ', ' + result['entity']['text']
            
        print(context)
        # Build a prompt with template for RAG
        # prompt_template = f"This is a prompt for a RAG system. I need you to create a fairy tale in catalan following the user prompt and the context. The prompt is the user prompt: {userPrompt} and this is the context: {context}. So the output should be in catalan."


        # llm_result = llm.invoke(prompt_template)

        # print(llm_result)
    except ConnectionError as e:
        print(e)
