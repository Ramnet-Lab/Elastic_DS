#!/usr/bin/env python3
"""
Script to combine Zeek connection log files.
Combines all files with 'conn.' in the name but excludes any summary files.
"""

import os
import glob
import pandas as pd

def combine_zeek_conn_logs(input_dir, output_file):
    """
    Combine all Zeek connection log files in the input directory.
    
    Args:
        input_dir: Directory containing Zeek conn log files
        output_file: Path to write the combined output file
    """
    # Get only the specific conn files with the pattern conn.YYYY-MM-DD-HH-MM-SS.csv
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    conn_files = [f for f in all_files if os.path.basename(f).startswith("conn.") and
                 len(os.path.basename(f).split(".")) == 3 and
                 "-" in os.path.basename(f).split(".")[1] and
                 "conn-summary" not in os.path.basename(f)]
    
    if not conn_files:
        print(f"No conn files found in {input_dir}")
        return
    
    print(f"Found {len(conn_files)} conn files to combine")
    
    # Read and combine files
    all_data = []
    header = None
    
    for i, file_path in enumerate(sorted(conn_files)):
        print(f"Processing file {i+1}/{len(conn_files)}: {os.path.basename(file_path)}")
        
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Store header from first file
        if header is None:
            header = lines[0]
            all_data.append(header)
        
        # Add data lines (skip header)
        all_data.extend(lines[1:])
    
    # Write combined data to output file
    with open(output_file, 'w') as f:
        f.writelines(all_data)
    
    print(f"Combined data written to {output_file}")
    print(f"Total lines: {len(all_data)}")

if __name__ == "__main__":
    input_dir = "zeek_data"
    output_file = "zeek_data/combined_conn_logs.csv"
    combine_zeek_conn_logs(input_dir, output_file)