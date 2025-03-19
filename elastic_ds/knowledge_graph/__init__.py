"""
Knowledge graph module for the Elastic Data Science Pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional


# Custom JSON encoder to handle Pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def build_knowledge_graph(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Any]:
    """
    Build a knowledge graph from the DataFrame.

    Args:
        df: Input DataFrame
        config: Knowledge graph configuration

    Returns:
        Graph model if successful, None otherwise
    """
    if not config.get("enabled", True):
        print("Knowledge graph is disabled in configuration")
        return None

    try:
        from .modeling import create_graph_model
        from .ingest import ingest_to_neo4j
        
        # Create graph model
        model = create_graph_model(df, config)
        if model is None:
            print("Failed to create graph model")
            return None
            
        # Ingest data to Neo4j if configured
        if config.get("ingest_to_neo4j", True):
            success = ingest_to_neo4j(df, model, config)
            if not success:
                print("Warning: Failed to ingest data to Neo4j")
        
        return model
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return None