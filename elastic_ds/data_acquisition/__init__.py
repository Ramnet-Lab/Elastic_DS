"""
Data acquisition module for the Elastic Data Science Pipeline.
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from .elasticsearch import acquire_from_elasticsearch
from .local import acquire_from_local
from .api import acquire_from_api


def acquire_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Acquire data from the specified source(s).

    Args:
        config: Data source configuration

    Returns:
        DataFrame with acquired data if successful, None otherwise
    """
    source_type = config.get("type", "local")
    
    # Check if we're using multiple sources
    if source_type == "multiple":
        return acquire_from_multiple_sources(config.get("sources", []))
    
    # Single source
    if source_type == "elasticsearch":
        return acquire_from_elasticsearch(config.get("elasticsearch", {}))
    elif source_type == "local":
        return acquire_from_local(config.get("local", {}))
    elif source_type == "api":
        return acquire_from_api(config.get("api", {}))
    else:
        print(f"Unknown data source type: {source_type}")
        return None


def acquire_from_multiple_sources(sources: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    Acquire data from multiple sources and combine them.

    Args:
        sources: List of source configurations

    Returns:
        Combined DataFrame if successful, None otherwise
    """
    if not sources:
        print("No sources specified")
        return None
    
    dfs = []
    
    for source in sources:
        source_type = source.get("type")
        if not source_type:
            print("Source type not specified, skipping")
            continue
        
        # Create a config with this source as the main source
        config = {"type": source_type}
        if source_type == "elasticsearch":
            config["elasticsearch"] = source.get("config", {})
        elif source_type == "local":
            config["local"] = source.get("config", {})
        elif source_type == "api":
            config["api"] = source.get("config", {})
        else:
            print(f"Unknown source type: {source_type}, skipping")
            continue
        
        # Acquire data from this source
        df = acquire_data(config)
        if df is not None:
            print(f"Acquired {len(df)} rows from {source_type} source")
            dfs.append(df)
    
    # Combine all DataFrames
    if not dfs:
        print("No data acquired from any source")
        return None
    
    if len(dfs) == 1:
        return dfs[0]
    else:
        # Combine DataFrames (this assumes they can be concatenated)
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Combined {len(combined_df)} rows from {len(dfs)} sources")
            return combined_df
        except Exception as e:
            print(f"Error combining DataFrames: {e}")
            # Return the first DataFrame as a fallback
            print(f"Returning {len(dfs[0])} rows from the first source")
            return dfs[0]