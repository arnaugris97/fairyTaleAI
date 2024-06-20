# fairytaleAI/inference_utils.py
import torch
from tokenizer.wordPieceTokenizer import WordPieceTokenizer
from BERT.BERT_model import BERT

def adjust_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'bert.' prefix from keys
        new_key = key.replace('bert.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def load_model(model_path, tokenizer_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path)
        
    # Initialize the tokenizer
    tokenizer = WordPieceTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Initialize the model with the loaded configuration
    model = BERT(
        vocab_size=tokenizer.vocab_size,
        d_model=checkpoint['BERT_hidden_size'],
        n_layers=checkpoint['BERT_num_hidden_layers'],
        heads=checkpoint['BERT_att_heads'],
        dropout=0.1  # Adjust if you have this value stored in the checkpoint
    )
    
    # Adjust state dict keys
    adjusted_state_dict = adjust_state_dict_keys(checkpoint['model_state_dict'])

    # Filter out the keys that are not part of the BERT model
    filtered_state_dict = {k: v for k, v in adjusted_state_dict.items() if k in model.state_dict()}
    
    # Load the model state
    model.load_state_dict(filtered_state_dict)
    model.eval()  # Set the model to evaluation mode
    
    return model, tokenizer

def preprocess_input(text, tokenizer, max_length=512):
    # Ensure text is a string
    text = str(text)
    
    # Tokenize and encode the text
    token_ids = tokenizer.encode(text)
    
    # Ensure the token_ids length is exactly max_length
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids = token_ids + [tokenizer.word2idx[tokenizer.pad_token]] * (max_length - len(token_ids))
    
    attention_mask = [1 if i < len(token_ids) else 0 for i in range(max_length)]
    token_type_ids = [0] * max_length
    
    # Convert lists to tensors
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }



def generate_embeddings(inputs, model):
    # Assuming inputs is a dictionary with input_ids and token_type_ids
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    
    # Call the model's forward method with the correct arguments
    outputs = model(input_ids, token_type_ids)
    
    # Detach the tensor from the computation graph and convert to NumPy array
    return outputs.detach().cpu()


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
