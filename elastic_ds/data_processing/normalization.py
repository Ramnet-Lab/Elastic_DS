"""
Normalization module for the Elastic Data Science Pipeline.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Tuple, List, Optional


def normalize_dataframe(df: pd.DataFrame, visualization_ready: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create a normalized version of the DataFrame.

    Args:
        df: Input DataFrame
        visualization_ready: Whether to prepare the data for visualization

    Returns:
        Tuple of (normalized DataFrame, mapping dictionary)
    """
    # Make a copy of the data
    df_norm = df.copy()

    # Create a dictionary to hold mapping information for each column
    mapping_dict = {}

    # Iterate over each column and transform its values
    for col in df_norm.columns:
        # If column contains list values, convert them to strings
        if df_norm[col].apply(lambda x: isinstance(x, list)).any():
            df_norm[col] = df_norm[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        # Process datetime columns: convert to integer timestamps then normalize
        if pd.api.types.is_datetime64_any_dtype(df_norm[col]):
            series_int = df_norm[col].astype('int64')
            min_val = series_int.min()
            max_val = series_int.max()
            # Save mapping for reverse normalization
            mapping_dict[col] = {"type": "datetime", "min": int(min_val), "max": int(max_val)}
            df_norm[col] = normalize_numeric(series_int)
        
        # Process numeric columns
        elif pd.api.types.is_numeric_dtype(df_norm[col]):
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            mapping_dict[col] = {"type": "numeric", "min": min_val, "max": max_val}
            df_norm[col] = normalize_numeric(df_norm[col])
        
        # Process categorical/object columns
        else:
            df_norm[col] = df_norm[col].astype(str)
            codes, uniques = pd.factorize(df_norm[col])
            mapping = {}
            n = len(uniques)
            for i, val in enumerate(uniques):
                # Calculate normalized value for each unique category
                if n > 1:
                    norm_value = i / (n - 1)
                else:
                    norm_value = 0.0
                mapping[str(norm_value)] = val
            mapping_dict[col] = {"type": "categorical", "mapping": mapping}
            if n > 1:
                df_norm[col] = codes / (n - 1)
            else:
                df_norm[col] = 0.0

    # Additional processing for visualization if requested
    if visualization_ready:
        # Ensure all columns are numeric for visualization
        for col in df_norm.columns:
            if not pd.api.types.is_numeric_dtype(df_norm[col]):
                df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')
        
        # Fill NaN values with 0 for visualization
        df_norm = df_norm.fillna(0)

    return df_norm, mapping_dict


def denormalize_dataframe(df_norm: pd.DataFrame, mapping_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Denormalize a DataFrame using the mapping dictionary.

    Args:
        df_norm: Normalized DataFrame
        mapping_dict: Mapping dictionary from normalize_dataframe

    Returns:
        Denormalized DataFrame
    """
    # Make a copy of the normalized data
    df_denorm = df_norm.copy()

    # Iterate over each column and transform its values back
    for col, info in mapping_dict.items():
        if col not in df_denorm.columns:
            continue

        # Process datetime columns
        if info["type"] == "datetime":
            min_val = info["min"]
            max_val = info["max"]
            df_denorm[col] = df_denorm[col] * (max_val - min_val) + min_val
            df_denorm[col] = pd.to_datetime(df_denorm[col])
        
        # Process numeric columns
        elif info["type"] == "numeric":
            min_val = info["min"]
            max_val = info["max"]
            df_denorm[col] = df_denorm[col] * (max_val - min_val) + min_val
        
        # Process categorical columns
        elif info["type"] == "categorical":
            mapping = info["mapping"]
            # Convert normalized values to original categories
            df_denorm[col] = df_denorm[col].apply(
                lambda x: mapping.get(str(round(x, 6)), x) if pd.notna(x) else x
            )

    return df_denorm


def normalize_numeric(series: pd.Series) -> pd.Series:
    """
    Normalize numeric data using min-max scaling.

    Args:
        series: Numeric series to normalize

    Returns:
        Normalized series
    """
    # If the series is boolean, convert it to integer
    if pd.api.types.is_bool_dtype(series):
        series = series.astype(int)
    
    min_val = series.min()
    max_val = series.max()
    
    # Avoid division by zero
    if max_val - min_val != 0:
        return (series - min_val) / (max_val - min_val)
    else:
        return pd.Series(0.0, index=series.index)