"""
Embeddings module for the Elastic Data Science Pipeline.
"""

import numpy as np
from typing import List, Optional


def compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """
    Compute embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        model_name: Name of the sentence transformer model

    Returns:
        Array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Sentence Transformers is not installed. Please install it with: pip install sentence-transformers")
        return None

    try:
        # Initialize the model
        print(f"Initializing Sentence Transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Compute embeddings
        print("Computing embeddings (this may take a minute)...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        return np.array(embeddings)
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return None


def compute_query_embedding(query: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """
    Compute embedding for a query.

    Args:
        query: Query text
        model_name: Name of the sentence transformer model

    Returns:
        Query embedding
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Sentence Transformers is not installed. Please install it with: pip install sentence-transformers")
        return None

    try:
        # Initialize the model
        model = SentenceTransformer(model_name)
        
        # Compute embedding
        embedding = model.encode([query])[0]
        
        return np.array(embedding)
    except Exception as e:
        print(f"Error computing query embedding: {e}")
        return None