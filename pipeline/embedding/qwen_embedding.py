# qwen_embedding.py
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------
# 1. Modèles
# --------------------------------------------------
embedding_model_id = "Qwen/Qwen3-Embedding-0.6B"

tokenizer_embed = AutoTokenizer.from_pretrained(embedding_model_id)
model_embed = AutoModel.from_pretrained(embedding_model_id)
model_embed.eval()

# Cross-encoder pour reranking (reranker)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --------------------------------------------------
# 2. Embedding
# --------------------------------------------------
def embed_chunks_qwen3(chunks, batch_size=8):
    """Encode les textes en vecteurs 1024-dim normalisés."""
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = tokenizer_embed(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model_embed(**inputs)
            vecs = outputs.last_hidden_state[:, 0]  # token CLS
            normed = torch.nn.functional.normalize(vecs, p=2, dim=1)
            embeddings.extend(normed.cpu().numpy())
    return embeddings

# --------------------------------------------------
# 3. Index brut (NearestNeighbors de sklearn)
# --------------------------------------------------
def build_faiss_index(embeddings, metric="euclidean"):
    """Construit un index de recherche avec sklearn (remplace Faiss)."""
    X = np.asarray(embeddings, dtype="float32")
    index = NearestNeighbors(metric=metric, algorithm="brute")
    index.fit(X)
    return index

# --------------------------------------------------
# 4. Recherche + Reranking Cross-Encoder
# --------------------------------------------------
def retrieve_top_k_chunks(question, text_chunks, chunk_embeddings, index, top_k=3):
    """
    Recherche initiale (ANN), puis reranking via cross-encoder.
    """
    # Embed la question
    q_vec = embed_chunks_qwen3([question])[0].reshape(1, -1).astype("float32")
    
    # Recherche initiale sur l’index
    _, indices = index.kneighbors(q_vec, n_neighbors=min(20, len(text_chunks)))

    # Candidats extraits
    candidates = [text_chunks[i] for i in indices[0]]

    # Reranking avec CrossEncoder
    pairs = [(question, chunk) for chunk in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = [
        chunk for chunk, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    ]

    return reranked[:top_k]


"""
sample_chunks = [
        "Le budget total pour 2024 est de 1,8 M€.",
        "Tableau : Ventilation du budget 2024 …",
        "Les dépenses 2023 ont dépassé les prévisions à cause de la logistique.",
        "Objectif 2025 : stabiliser les coûts et investir dans l'innovation."
    ]
embs = embed_chunks_qwen3(sample_chunks)
idx = build_faiss_index(embs, metric="euclidean")
print(len(embs[0]))
print(embs[0])
q = "Pourquoi les dépenses ont augmenté en 2023 ?"
print("TOP CHUNKS →", retrieve_top_k_chunks(q, sample_chunks, embs, idx,1))
"""