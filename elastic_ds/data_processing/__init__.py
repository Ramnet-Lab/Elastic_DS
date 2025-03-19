"""
Data processing module for the Elastic Data Science Pipeline.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import os

from .normalization import normalize_dataframe, denormalize_dataframe
from .cleaning import clean_dataframe


def process_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Process the data according to the configuration.

    Args:
        df: Input DataFrame
        config: Processing configuration

    Returns:
        Tuple of (processed DataFrame, mapping dictionary)
    """
    # Create output directory if it doesn't exist
    output_dir = config.get("output_dir", "./processed_data")
    os.makedirs(output_dir, exist_ok=True)

    # Clean the data
    df_clean = clean_dataframe(df)

    # Save the cleaned data
    clean_csv = os.path.join(output_dir, "cleaned_data.csv")
    df_clean.to_csv(clean_csv, index=False, encoding="utf-8")
    print(f"Cleaned data exported to {clean_csv}")

    # Normalize the data if requested
    mapping_dict = None
    if config.get("normalize", True):
        df_norm, mapping_dict = normalize_dataframe(df_clean, 
                                                   visualization_ready=config.get("visualization_ready", True))
        
        # Save the normalized data
        normalized_csv = os.path.join(output_dir, "normalized_data.csv")
        df_norm.to_csv(normalized_csv, index=False, encoding="utf-8")
        print(f"Normalized data exported to {normalized_csv}")
        
        # Return the normalized DataFrame
        return df_norm, mapping_dict
    
    # Return the cleaned DataFrame if normalization is not requested
    return df_clean, mapping_dict