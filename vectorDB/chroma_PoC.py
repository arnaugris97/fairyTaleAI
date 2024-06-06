import chromadb
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

# Initialize Chroma DB
chroma = chromadb.Client()


# Create a collection to store embeddings
collection = chroma.create_collection(name="BERT_embeddings", metadata={"hnsw:space": "cosine"})

# Example embedding and text
example_documents = ["This is a sample sentence.", "Another example sentence.", "Yet another one.", "Yet another one."]

example_embeddings = [np.random.rand(768).tolist() for _ in example_documents]

# Add embeddings to Chroma DB
collection.add(
    documents=example_documents,
    embeddings=example_embeddings,
    metadatas={"title": "en"},
    ids=[f"id{i}" for i in range(len(example_documents))],
)

# Query with a new embedding
results = collection.query(query_embeddings=[example_embeddings[0]], n_results=1, include=["embeddings"])

# Retrieve the corresponding texts
print(f'ids: ',results["ids"])
print(f'documents: ',results["documents"])
print(f'distances: ',results["distances"])


embeddings = np.array(example_embeddings)

# Perform PCA for initial dimensionality reduction to 2 dimensions
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Create a DataFrame for visualization
df = pd.DataFrame(embeddings_2d, columns=['Component 1', 'Component 2'])
df['text'] = example_documents

# Plotting with Plotly
fig = px.scatter(df, x='Component 1', y='Component 2', text='text', title='PCA projection of embeddings')
fig.update_traces(textposition='top center')
fig.show()