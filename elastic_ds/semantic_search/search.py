"""
Search module for the Elastic Data Science Pipeline.
"""

import numpy as np
from typing import List, Tuple, Optional

from .embeddings import compute_query_embedding


def semantic_search(query: str, texts: List[str], embeddings: np.ndarray, top_n: int = 10) -> List[Tuple[float, str]]:
    """
    Perform semantic search using cosine similarity.

    Args:
        query: Query text
        texts: List of texts to search
        embeddings: Pre-computed embeddings for texts
        top_n: Number of top results to return

    Returns:
        List of (similarity score, text) tuples
    """
    # Compute query embedding
    query_embedding = compute_query_embedding(query)
    if query_embedding is None:
        print("Error: Failed to compute query embedding")
        return []

    # Compute cosine similarity
    similarities = compute_cosine_similarity(query_embedding, embeddings)
    
    # Get top N results
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Create result list
    results = []
    for idx in top_indices:
        sim_score = similarities[idx]
        text = texts[idx]
        results.append((sim_score, text))
    
    return results


def compute_cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query embedding and a set of embeddings.

    Args:
        query_embedding: Query embedding
        embeddings: Set of embeddings

    Returns:
        Array of cosine similarities
    """
    # Compute dot products
    dot_products = np.dot(embeddings, query_embedding)
    
    # Compute norms
    message_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    
    # Compute cosine similarities
    cosine_similarities = dot_products / (message_norms * query_norm)
    
    return cosine_similarities