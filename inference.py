from inference_utils import load_model, preprocess_input, generate_embeddings, search_chromadb, generate_output
import chromadb

if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint20Epochs.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path, tokenizer_path)
    
    user_input = "I want a fairy tale about a desert and a siren"
    inputs = preprocess_input(user_input, tokenizer)
    embeddings = generate_embeddings(inputs, model)
    
    # Initialize ChromaDB client
    chromadb_client = chromadb.Client()
    
    try:
        results = search_chromadb(embeddings, chromadb_client)
        output = generate_output(results)
        print(output)
    except ConnectionError as e:
        print(e)
