from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import numpy as np
from umap import UMAP
import plotly.express as px

ds = load_dataset("bowen-upenn/PersonaMem-v2")
model = SentenceTransformer('all-MiniLM-L6-v2')

test = 'Convertir cette phrase en embedding.'

def add_preference_embeddings_batch(batch):
    batch['preference_embedding'] = model.encode(batch['preference'])
    return batch

ds["benchmark_text"] = ds["benchmark_text"].map(
    add_preference_embeddings_batch, 
    batched=True, 
    batch_size=32
)

embeddings = np.array(ds["benchmark_text"]['preference_embedding'])

# Réduction de dimensionnalité à 3D avec UMAP
reducer = UMAP(n_components=3, n_jobs=-1)
embeddings_3d = reducer.fit_transform(embeddings)

# Visualisation interactive avec Plotly
fig = px.scatter_3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    color=ds["benchmark_text"]['pref_type'],  # Colorer par type de préférence
    hover_data=[ds["benchmark_text"]['preference']],  # Afficher la préférence au survol
    title="Visualisation 3D des embeddings de préférences"
)

fig.show()