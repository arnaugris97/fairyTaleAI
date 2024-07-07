from createTL_embedding_DB import MilvusEmbeddingProcessorTL
from langchain_community.llms import Ollama


if __name__ == "__main__":
    model_path = 'Checkpoints/TL100Epochs.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'

    processor = MilvusEmbeddingProcessorTL(model_path, '')
    llm = Ollama(model="llama3")

    userPrompt = "Once upon a time there was a little riding hood who lived in a small village."
    
    try:
        results = processor.process_query(userPrompt)
        context = ''
        for result in results[0]:
            for sentence in result:
            
                if context == '':
                    context += sentence['entity']['text']
                else:
                    context += ', ' + sentence['entity']['text']
            
        
        # Build a prompt with template for RAG
        prompt_template = f"This is a prompt for a RAG system. I need you to create a fairy tale in catalan following the user prompt and the context. The prompt is the user prompt: {userPrompt} and this is the context: {context}. So the output should be in catalan."


        llm_result = llm.invoke(prompt_template)

        print(llm_result)
    except ConnectionError as e:
        print(e)
