"""
Pipeline module for the Elastic Data Science Pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from .config import Config
from .data_acquisition import acquire_data
from .data_processing import process_data
from .knowledge_graph import build_knowledge_graph, CustomJSONEncoder
from .semantic_search import perform_semantic_search


def handle_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle mixed data types in a DataFrame to prevent warnings and ensure consistency.
    
    Args:
        df: Input DataFrame with potentially mixed types
        
    Returns:
        DataFrame with consistent data types
    """
    # Convert object columns that should be numeric
    for col in df.select_dtypes(include=['object']).columns:
        # Try to convert to numeric, coercing errors to NaN
        numeric_data = pd.to_numeric(df[col], errors='coerce')
        
        # If the conversion was mostly successful (few NaNs), use the numeric version
        if numeric_data.notna().sum() > 0.8 * len(numeric_data):
            df[col] = numeric_data
    
    # Convert datetime columns
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower() or 'ts' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    return df


class Pipeline:
    """Main pipeline for the Elastic Data Science Pipeline."""

    def __init__(self, config: Optional[Config] = None, config_file: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            config: Configuration object
            config_file: Path to configuration file
        """
        if config is None:
            self.config = Config(config_file=config_file)
        else:
            self.config = config
        
        # Create output directory
        self.output_dir = self.config.get("output_dir", "./output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data
        self.df = None
        self.df_processed = None
        self.mapping_dict = None
        self.graph_model = None
        self.search_results = None

    def run(self) -> bool:
        """
        Run the pipeline.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Acquire data or use existing data
            print("\n=== Step 1: Data Acquisition ===")
            
            # Check if we should use existing data
            use_existing_data = self.config.get("data_acquisition", {}).get("use_existing_data", False)
            raw_csv = os.path.join(self.output_dir, "raw_data.csv")
            
            if use_existing_data and os.path.exists(raw_csv):
                print("Using existing data from previous run...")
                try:
                    # Use low_memory=False to handle mixed data types and prevent warnings
                    self.df = pd.read_csv(raw_csv, low_memory=False)
                    # Apply data type handling to fix mixed types
                    self.df = handle_mixed_types(self.df)
                    print(f"Loaded {len(self.df)} rows of data from {raw_csv}")
                except Exception as e:
                    print(f"Error loading existing data: {e}")
                    print("Falling back to acquiring new data...")
                    use_existing_data = False
            
            # If not using existing data or loading failed, acquire new data
            if not use_existing_data or self.df is None:
                print("Acquiring new data...")
                self.df = acquire_data(self.config.get("data_source", {}))
                if self.df is None:
                    print("Error: Failed to acquire data")
                    return False
                print(f"Acquired {len(self.df)} rows of data")
                
                # Apply data type handling before saving
                self.df = handle_mixed_types(self.df)
                # Save raw data with proper type preservation
                self.df.to_csv(raw_csv, index=False, encoding="utf-8")
                print(f"Raw data saved to {raw_csv}")
            
            # Step 2: Process data
            print("\n=== Step 2: Processing Data ===")
            self.df_processed, self.mapping_dict = process_data(self.df, self.config.get("processing", {}))
            if self.df_processed is None:
                print("Error: Failed to process data")
                return False
            print(f"Processed {len(self.df_processed)} rows of data")
            
            # Step 3: Build knowledge graph (if enabled)
            if self.config.get("knowledge_graph", {}).get("enabled", True):
                print("\n=== Step 3: Building Knowledge Graph ===")
                
                # Prepare data for knowledge graph
                df_for_kg = self.df.copy()
                
                # Convert datetime columns to strings
                for col in df_for_kg.select_dtypes(include=['datetime64']).columns:
                    df_for_kg[col] = df_for_kg[col].astype(str)
                
                # Handle NaN values
                df_for_kg = df_for_kg.fillna("UNKNOWN")
                
                # Convert any other problematic types
                for col in df_for_kg.columns:
                    if df_for_kg[col].dtype.name == 'object':
                        # Try to convert objects to strings if they're not already
                        try:
                            df_for_kg[col] = df_for_kg[col].astype(str)
                        except:
                            pass
                
                # Set knowledge graph output directory
                kg_config = self.config.get("knowledge_graph", {}).copy()
                kg_config["output_dir"] = os.path.join(self.output_dir, "kg_output")
                os.makedirs(kg_config["output_dir"], exist_ok=True)
                
                self.graph_model = build_knowledge_graph(df_for_kg, kg_config)
                if self.graph_model is None:
                    print("Warning: Failed to build knowledge graph")
            else:
                print("\n=== Step 3: Knowledge Graph Disabled ===")
            
            # Step 4: Perform semantic search (if enabled)
            if self.config.get("semantic_search", {}).get("enabled", True):
                print("\n=== Step 4: Performing Semantic Search ===")
                self.search_results = perform_semantic_search(self.df, self.config.get("semantic_search", {}))
                if self.search_results is None:
                    print("Warning: Failed to perform semantic search")
            else:
                print("\n=== Step 4: Semantic Search Disabled ===")
            
            print("\n=== Pipeline Completed Successfully ===")
            return True
        
        except Exception as e:
            print(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the pipeline.

        Returns:
            Dictionary with pipeline results
        """
        return {
            "raw_data": self.df,
            "processed_data": self.df_processed,
            "mapping_dict": self.mapping_dict,
            "graph_model": self.graph_model,
            "search_results": self.search_results
        }