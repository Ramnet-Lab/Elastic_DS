"""
Graph visualization module for the Elastic Data Science Pipeline.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, Optional


def visualize_graph(model: Any, output_dir: str) -> None:
    """
    Visualize a graph model.

    Args:
        model: Graph model
        output_dir: Output directory for visualization
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in getattr(model, 'nodes', []):
        node_id = normalize_label(node)
        G.add_node(node_id)
    
    # Add edges
    if hasattr(model, 'relationships'):
        for rel in model.relationships:
            src_obj = getattr(rel, 'source', getattr(rel, 'start', None))
            dst_obj = getattr(rel, 'target', getattr(rel, 'end', None))
            if src_obj and dst_obj:
                src_id = normalize_label(src_obj)
                dst_id = normalize_label(dst_obj)
                G.add_edge(src_id, dst_id)
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # For reproducibility
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=15, width=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
    
    # Add title
    plt.title("Knowledge Graph Model Visualization", fontsize=16)
    
    # Remove axis
    plt.axis("off")
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "knowledge_graph.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Graph visualization saved to {os.path.join(output_dir, 'knowledge_graph.png')}")


def normalize_label(node_obj) -> str:
    """
    Convert a node object into a consistent, hashable identifier.

    Args:
        node_obj: Node object

    Returns:
        Normalized label
    """
    # If there's a unique 'id' or 'name' attribute, use that
    if hasattr(node_obj, 'id'):
        return str(node_obj.id)  # Ensure it's a string
    
    if hasattr(node_obj, 'name'):
        return str(node_obj.name)
    
    # Fallback: parse the string representation, e.g. '(:Label)'
    s = str(node_obj)
    if s.startswith('(:') and s.endswith(')'):
        # Strip off the '(:' and the closing ')'
        return s[2:-1].strip()
    
    return s