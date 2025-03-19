"""
Data cleaning module for the Elastic Data Science Pipeline.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by handling missing values, converting types, etc.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Make a copy of the data
    df_clean = df.copy()

    # Handle different data types
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # Replace NaN in numeric columns with 0
            df_clean[col] = df_clean[col].apply(
                lambda x: 0 if pd.isna(x) or (isinstance(x, float) and np.isnan(x)) or x == "" else x
            )
        elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            # Replace NaN in datetime columns with epoch start
            df_clean[col] = df_clean[col].fillna(pd.Timestamp("1970-01-01"))
        else:
            # Replace NaN in string/object columns with "UNKNOWN"
            df_clean[col] = df_clean[col].replace({None: "UNKNOWN", pd.NA: "UNKNOWN"}).fillna("UNKNOWN")
            df_clean[col] = df_clean[col].astype(str).apply(
                lambda x: "UNKNOWN" if x.lower() in ["nan", "none", "null", ""] else x
            )

    # Fill any remaining NaN values with "UNKNOWN"
    df_clean.fillna("UNKNOWN", inplace=True)

    # Replace empty strings with "UNKNOWN"
    df_clean.replace(r'^\s*$', "UNKNOWN", regex=True, inplace=True)

    return df_clean


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to ensure they are valid for various operations.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned column names
    """
    # Make a copy of the data
    df_clean = df.copy()

    # Clean column names
    df_clean.columns = [
        re.sub(r'[^\w\s]', '_', col)  # Replace special characters with underscore
        .strip()  # Remove leading/trailing whitespace
        .lower()  # Convert to lowercase
        .replace(' ', '_')  # Replace spaces with underscore
        for col in df_clean.columns
    ]

    return df_clean


def generate_column_mapping(columns: list) -> Dict[str, str]:
    """
    Generate a mapping from original column names to camelCase names.

    Args:
        columns: List of column names

    Returns:
        Dictionary mapping original names to camelCase names
    """
    mapping = {}
    for col in columns:
        # Split by non-alphanumeric characters
        parts = re.split(r'[^a-zA-Z0-9]', col)
        # Filter out empty parts
        parts = [part for part in parts if part]
        if not parts:
            mapping[col] = col
            continue
        
        # Convert to camelCase
        camel_case = parts[0].lower()
        for part in parts[1:]:
            if part:
                camel_case += part[0].upper() + part[1:].lower()
        
        mapping[col] = camel_case
    
    return mapping