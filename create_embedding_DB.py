import pandas as pd
import chromadb
from inference_utils import load_model, preprocess_input, generate_embeddings
from tokenizer.wordPieceTokenizer import WordPieceTokenizer

class ChromaEmbeddingProcessor:
    def __init__(self, model_path, tokenizer_path, csv_file):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.csv_file = csv_file
        self.chromadb_client = chromadb.Client()
        self.model, self.tokenizer = self.load_bert_model()  # Load BERT model and tokenizer
        self.model.eval()  # Set the model to evaluation mode

    def load_bert_model(self):
        # Initialize model and tokenizer
        return load_model(self.model_path, self.tokenizer_path)

    def load_sentences_from_csv(self):
        df = pd.read_csv(self.csv_file)
        sentences = df['Sentence'].tolist()
        titles = df['Title'].tolist()
        return sentences, titles

    def process_and_store_embeddings(self):
        sentences, titles = self.load_sentences_from_csv()
        embeddings = []
        num_sentences = len(sentences)
        
        for i, (sentence, title) in enumerate(zip(sentences, titles)):
            # Ensure sentence is a string
            sentence = str(sentence)
            inputs = preprocess_input(sentence, self.tokenizer)
            print("Inputs are processed")
            sentence_embeddings = generate_embeddings(inputs, self.model).detach().numpy().tolist()
            print("Sentences are embedded")
            embeddings.append((sentence_embeddings, sentence, title))
            
            # Calculate and print the percentage completion
            percent_complete = ((i + 1) / num_sentences) * 100
            print(f"Processed {i + 1}/{num_sentences} sentences. {percent_complete:.2f}% complete")

        collection_name = 'BERT_embeddings'
        for embedding, sentence, title in embeddings:
            try:
                result = self.chromadb_client.get_collection(name=collection_name).insert_one(
                    {
                        'embedding': embedding,
                        'sentence': sentence,
                        'title': title
                    }
                )
                print(f"Inserted document with _id: {result.inserted_id}")
            except Exception as e:
                print(f"Failed to insert document: {e}")



# Example usage:
if __name__ == "__main__":
    model_path = 'Checkpoints/checkpoint.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'
    csv_file = 'dataset/dataset_sentences.csv'

    processor = ChromaEmbeddingProcessor(model_path, tokenizer_path, csv_file)
    processor.process_and_store_embeddings()
