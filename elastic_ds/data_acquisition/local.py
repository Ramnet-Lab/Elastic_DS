"""
Local file connector for the Elastic Data Science Pipeline.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional


def acquire_from_local(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Acquire data from a local file.

    Args:
        config: Local file configuration

    Returns:
        DataFrame with data from the local file if successful, None otherwise
    """
    try:
        file_path = config.get("file_path", "output.csv")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        elif ext == '.parquet':
            return pd.read_parquet(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return None
            
    except Exception as e:
        print(f"Error acquiring data from local file: {e}")
        import traceback
        traceback.print_exc()
        return None