"""
API connector for the Elastic Data Science Pipeline.
"""

import os
import requests
import pandas as pd
import subprocess
import sys
from typing import Dict, Any, List, Optional


def acquire_from_api(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Acquire data from an API.

    Args:
        config: API configuration

    Returns:
        DataFrame with data from the API if successful, None otherwise
    """
    try:
        base_url = config.get("base_url", "http://localhost:5000/files")
        save_dir = config.get("save_dir", "csv_downloads")
        
        # List available CSV files
        response = requests.get(base_url)
        if response.status_code != 200:
            print(f"Error listing files: {response.status_code}, {response.text}")
            return None
            
        csv_files = response.json()
        if not csv_files:
            print("No CSV files available from the API")
            return None
        
        # Download only conn files (not summary files)
        downloaded_files = []
        for filename in csv_files:
            try:
                # Filter to include only conn files and exclude summary files
                if filename.startswith("conn.") and "conn-summary" not in filename:
                    # Create directory if it doesn't exist
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Download file
                    file_url = f"{base_url}/{filename}"
                    response = requests.get(file_url, stream=True)
                    
                    if response.status_code == 200:
                        file_path = os.path.join(save_dir, filename)
                        with open(file_path, 'wb') as file:
                            for chunk in response.iter_content(chunk_size=1024):
                                file.write(chunk)
                        downloaded_files.append(file_path)
                        print(f"Downloaded: {file_path}")
                    else:
                        print(f"Failed to download {filename}: {response.status_code}")
                else:
                    print(f"Skipping file (not a conn file or is a summary file): {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        
        # Use the combine_zeek_conn_logs.py script to combine the files
        print("Combining Zeek connection log files...")
        combined_file = os.path.join(save_dir, "combined_conn_logs.csv")
        
        try:
            # Check if the combine script exists
            script_path = os.path.join(os.getcwd(), "combine_zeek_conn_logs.py")
            if os.path.exists(script_path):
                # Run the combine script
                result = subprocess.run([sys.executable, script_path],
                                       capture_output=True, text=True, check=True)
                print(result.stdout)
                
                # Read the combined file
                if os.path.exists(combined_file):
                    print(f"Reading combined file: {combined_file}")
                    return pd.read_csv(combined_file)
                else:
                    print(f"Combined file not found: {combined_file}")
            else:
                print(f"Combine script not found: {script_path}")
        except Exception as e:
            print(f"Error running combine script: {e}")
            print("Falling back to manual combination...")
        
        # Fallback: Combine all downloaded files into a single DataFrame manually
        print("Manually combining downloaded files...")
        dfs = []
        for file_path in downloaded_files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if not dfs:
            print("No data found in downloaded CSV files")
            return None
        
        # Combine all DataFrames
        if len(dfs) == 1:
            return dfs[0]
        else:
            # This assumes all DataFrames have the same schema
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save the manually combined data
            if not os.path.exists(combined_file):
                combined_df.to_csv(combined_file, index=False)
                print(f"Manually combined data saved to {combined_file}")
                
            return combined_df
            
    except Exception as e:
        print(f"Error acquiring data from API: {e}")
        import traceback
        traceback.print_exc()
        return None