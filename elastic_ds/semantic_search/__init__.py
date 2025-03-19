"""
Semantic search module for the Elastic Data Science Pipeline.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import os

from .embeddings import compute_embeddings
from .search import semantic_search


def perform_semantic_search(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Perform semantic search on the DataFrame.

    Args:
        df: Input DataFrame
        config: Semantic search configuration

    Returns:
        Dictionary with search results
    """
    if not config.get("enabled", True):
        print("Semantic search is disabled in configuration")
        return None

    # Check if the DataFrame has a 'message' column
    if "message" not in df.columns:
        print("Error: 'message' column not found in the DataFrame")
        return None

    # Ensure messages are strings and remove missing values
    messages = df["message"].dropna().astype(str).tolist()
    if not messages:
        print("Error: No valid messages found in the DataFrame")
        return None

    # Compute embeddings for messages
    model_name = config.get("model", "all-MiniLM-L6-v2")
    message_embeddings = compute_embeddings(messages, model_name)
    if message_embeddings is None:
        print("Error: Failed to compute embeddings")
        return None

    # Get the query from configuration
    query = config.get("default_query", "login authentication failure sudo user")
    top_n = config.get("top_n", 10)

    # Perform semantic search
    results = semantic_search(query, messages, message_embeddings, top_n)
    
    # Create output directory if it doesn't exist
    output_dir = config.get("output_dir", "./search_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to a file
    results_file = os.path.join(output_dir, "search_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Query: {query}\n\n")
        for i, (score, message) in enumerate(results, 1):
            f.write(f"{i}. Similarity: {score:.4f} | Message: {message}\n\n")
    
    print(f"Search results saved to {results_file}")
    
    return {
        "query": query,
        "results": results
    }