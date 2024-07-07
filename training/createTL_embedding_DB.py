import pandas as pd
from pymilvus import MilvusClient

from utils.inference_utils import generate_TL_embeddings, load_model_TL, preprocess_input
from torch.utils.data import DataLoader
from dataloader.custom_dataset import Custom_Inference_TL_Dataset_DB
from transformers import DistilBertTokenizer

class MilvusEmbeddingProcessorTL:
    def __init__(self, model_path, csv_file):
        self.model_path = model_path
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.csv_file = csv_file
        self.collection_name = "BERT_embeddings_trained"
        self.model = self.load_bert_model()
        self.model.eval()

        # Connect to Milvus
        self.client = MilvusClient(
            uri="https://in03-38809e18a6b7a16.api.gcp-us-west1.zillizcloud.com",
            token="db_38809e18a6b7a16:Ld8)J(@h1EUYw..)",
        )

    def load_bert_model(self):
        return load_model_TL(self.model_path, self.tokenizer)

       
    def load_sentences_from_csv(self):
        return pd.read_csv(self.csv_file)
    

    def process_and_store_embeddings(self):
        dataset = Custom_Inference_TL_Dataset_DB(self.load_sentences_from_csv(), self.tokenizer, 512)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        id_counter = 0
        
        for step, data in enumerate(dataloader):
            sentence_embeddings = generate_TL_embeddings(data, self.model)  # Generate embeddings
            embeddings = [sentence_embeddings[i].tolist() for i in range(len(sentence_embeddings))]
            texts = [data['sentence'][i] for i in range(len(sentence_embeddings))]
            titles = [data['title'][i] for i in range(len(sentence_embeddings))]

            entities = [
                {"id": id_counter + i, "embedding": embedding, "text": text, "title": title}
                for i, (embedding, title, text) in enumerate(zip(embeddings, titles, texts))
            ]
            
            id_counter += len(entities)

            self.client.insert(collection_name=self.collection_name, data=entities)
            print(f"Inserted batch {step+1} into Milvus from {len(dataset)/64}.")
            print("Percentage: ", (step+1)/(len(dataset)/64)*100, "%.")

        print("All embeddings inserted successfully.")

    def query_embeddings(self, query_embedding, top_k=5):
        results = self.client.search(collection_name=self.collection_name, data=[query_embedding], limit=top_k, output_fields=["text", "title"])
        return results

    def process_query(self, sentence):
        inputs = preprocess_input(str(sentence), self.tokenizer, isBERT=True)
        sentence_embeddings = generate_TL_embeddings(inputs, self.model).squeeze().tolist()
        results = self.query_embeddings(sentence_embeddings, 10)
        return results, inputs['input_ids']

# Example usage:
if __name__ == "__main__":
    model_path = 'Checkpoints/TL100Epochs.pt'
    csv_file = 'dataset/dataset_sentences_cleaned.csv'

    processor = MilvusEmbeddingProcessorTL(model_path, csv_file)
    processor.process_and_store_embeddings()

    # Only for testing purposes
    # results = processor.process_query('A story of flying dragons')
    # for result in results[0]:
    #     print(result)
