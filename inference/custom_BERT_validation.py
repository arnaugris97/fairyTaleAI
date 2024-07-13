

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from pymilvus import MilvusClient
from sklearn.model_selection import train_test_split
from utils.inference_utils import generate_TL_embeddings, generate_embeddings, load_model, preprocess_input
from transformers import DistilBertTokenizer



class Custom_BERT_validation:
    def __init__(self, model_path, tokenizer_path, csv_file):
        self.model_path = model_path
        self.csv_file = csv_file
        self.collection_name = 'BERT_embeddings'
        self.tokenizer_path = tokenizer_path
        self.test_data = self.load_testset()
        self.model, self.tokenizer = self.load_bert_model()
        self.model.eval()

        # Connect to Milvus
        self.client = MilvusClient(
            uri="https://in03-38809e18a6b7a16.api.gcp-us-west1.zillizcloud.com",
            token="db_38809e18a6b7a16:Ld8)J(@h1EUYw..)",
        )


    def load_bert_model(self):
        return load_model(self.model_path, self.tokenizer_path)


    def load_testset(self):
        dataset_csv = pd.read_csv(self.csv_file)
        random_state = 123  # We will be testing the Test set defined during training
        train, val_test = train_test_split(dataset_csv, test_size=0.2, random_state=random_state)
        val, test = train_test_split(val_test, test_size=0.5, random_state=random_state)
        return test
    
    def query_embeddings(self, query_embedding, top_k=5):
        results = self.client.search(collection_name=self.collection_name, data=[query_embedding], limit=top_k, output_fields=["text", "title"])
        return results
    
    def process_and_retrieve_similar_embeddings(self, topk, nsamples=100):
         # Sample 100 random rows from the test data
        random_test_data = self.test_data.sample(n=nsamples, random_state=42)
        accuracy_counter = 0
        for index, row in random_test_data.iterrows():
            inputs = preprocess_input(str(row['Sentence']), self.tokenizer)
            query_embedding = generate_embeddings(inputs, self.model).squeeze().tolist()
            results = self.query_embeddings(query_embedding, topk)
            # Check if title is in results[0]
            for result in results[0]:
                
                if row['Title'] == result['entity']['title']:
                    accuracy_counter += 1
                    print('Match found!', accuracy_counter)
    
            # print(f"Index: {index}, Title: {row['Title']}, Sentence: {row['Sentence']}")

        print("Accuracy: ", accuracy_counter/nsamples*100, "%.")
        


if __name__ == "__main__":
    model_path = 'Checkpoints/GoodCheckpoint.pt'
    tokenizer_path = 'tokenizer/wordPieceVocab.json'
    csv_file = 'dataset/dataset_sentences_cleaned.csv'

    processor = Custom_BERT_validation(model_path, tokenizer_path, csv_file)
    processor.process_and_retrieve_similar_embeddings(5, 100)
    