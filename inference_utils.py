# fairytaleAI/inference_utils.py
import torch
from tokenizer.wordPieceTokenizer import WordPieceTokenizer
from BERT.BERT_model import BERT

def load_model(model_path, tokenizer_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    
    # Extract configuration parameters from the checkpoint
    config = checkpoint["config"]  # Extract the configuration from the checkpoint
    
    # Initialize the tokenizer
    tokenizer = WordPieceTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Initialize the model with the loaded configuration
    model = BERT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        hidden_size=config['BERT_hidden_size'],
        segment_vocab_size=config['segment_vocab_size'],
        num_hidden_layers=config['BERT_num_hidden_layers'],
        num_attention_heads=config['BERT_att_heads'],
        intermediate_size=config['intermediate_size_multiplier'] * config['BERT_hidden_size'],
        batch_size=64  # This can be a default value or loaded from the config if needed
    )
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    
    return model, tokenizer

def preprocess_input(text, tokenizer, max_length=512):
    # Encode the input text using the custom tokenizer
    token_ids = tokenizer.encode(text)
    
    # Add special tokens and create attention mask and token type ids
    padded_token_ids, attention_mask, token_type_ids = tokenizer.add_special_tokens(token_ids, [], max_length=max_length)
    
    # Convert lists to tensors
    input_ids = torch.tensor([padded_token_ids])
    attention_mask = torch.tensor([attention_mask])
    token_type_ids = torch.tensor([token_type_ids])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }

def generate_embeddings(inputs, model):
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
    embeddings = outputs[0].mean(dim=1)
    return embeddings

def search_chromadb(embedding, chromadb_client, top_k=5):
    try:
        embedding_np = embedding.numpy().tolist()
        results = chromadb_client.get_collection(name="BERT_embeddings").query(query_embeddings=[embedding_np], n_results=top_k)
        return results
    except Exception as e:
        raise ConnectionError("Connection to Chroma not found") from e

def generate_output(results):
    output = [result for result in results["documents"]]
    return output
