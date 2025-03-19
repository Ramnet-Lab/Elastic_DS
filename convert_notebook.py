"""
Script to convert an existing notebook to use the Elastic Data Science Pipeline.
"""

import os
import pandas as pd
from elastic_ds.config import Config
from elastic_ds.pipeline import Pipeline


def convert_els_pull():
    """Convert ELS_pull.ipynb to use the pipeline."""
    print("Converting ELS_pull.ipynb to use the Elastic Data Science Pipeline")
    
    # Create configuration for Elasticsearch data source
    config = Config({
        "data_source": {
            "type": "elasticsearch",
            "elasticsearch": {
                "host": "http://192.168.2.46:9200",  # Remote Elasticsearch IP from ELS_pull.ipynb
                "index_pattern": "*",
                "verify_certs": False,
                "timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            }
        },
        "processing": {
            "normalize": False,  # Don't normalize for this example
            "output_dir": "./els_pull_output"
        },
        "knowledge_graph": {
            "enabled": False  # Disable knowledge graph for this example
        },
        "semantic_search": {
            "enabled": False  # Disable semantic search for this example
        },
        "output_dir": "./els_pull_output"
    })
    
    # Create and run pipeline
    pipeline = Pipeline(config)
    success = pipeline.run()
    
    if success:
        print("Pipeline completed successfully!")
        
        # Get results
        results = pipeline.get_results()
        
        # Print some information about the results
        if results["raw_data"] is not None:
            print(f"Raw data shape: {results['raw_data'].shape}")
            
            # Save the data to output.csv (as in the original notebook)
            output_csv = os.path.join(config.get("output_dir"), "output.csv")
            results["raw_data"].to_csv(output_csv, index=False, encoding="utf-8")
            print(f"Data exported to {output_csv}")
    else:
        print("Pipeline failed!")


def convert_els_norm():
    """Convert ELS_norm.ipynb to use the pipeline."""
    print("Converting ELS_norm.ipynb to use the Elastic Data Science Pipeline")
    
    # Check if output.csv exists
    if not os.path.exists("output.csv"):
        print("Error: output.csv not found. Run convert_els_pull() first.")
        return
    
    # Create configuration for local data source
    config = Config({
        "data_source": {
            "type": "local",
            "local": {
                "file_path": "output.csv"
            }
        },
        "processing": {
            "normalize": True,  # Enable normalization
            "output_dir": "./els_norm_output",
            "visualization_ready": True
        },
        "knowledge_graph": {
            "enabled": False  # Disable knowledge graph for this example
        },
        "semantic_search": {
            "enabled": False  # Disable semantic search for this example
        },
        "output_dir": "./els_norm_output"
    })
    
    # Create and run pipeline
    pipeline = Pipeline(config)
    success = pipeline.run()
    
    if success:
        print("Pipeline completed successfully!")
        
        # Get results
        results = pipeline.get_results()
        
        # Print some information about the results
        if results["processed_data"] is not None:
            print(f"Processed data shape: {results['processed_data'].shape}")
    else:
        print("Pipeline failed!")


def convert_semantic_search():
    """Convert Sematic_search.ipynb to use the pipeline."""
    print("Converting Sematic_search.ipynb to use the Elastic Data Science Pipeline")
    
    # Check if output.csv exists
    if not os.path.exists("output.csv"):
        print("Error: output.csv not found. Run convert_els_pull() first.")
        return
    
    # Create configuration for local data source
    config = Config({
        "data_source": {
            "type": "local",
            "local": {
                "file_path": "output.csv"
            }
        },
        "processing": {
            "normalize": False,  # Don't normalize for semantic search
            "output_dir": "./semantic_search_output"
        },
        "knowledge_graph": {
            "enabled": False  # Disable knowledge graph for this example
        },
        "semantic_search": {
            "enabled": True,
            "model": "all-MiniLM-L6-v2",  # From Sematic_search.ipynb
            "default_query": "login authentication failure sudo user",  # From Sematic_search.ipynb
            "top_n": 10
        },
        "output_dir": "./semantic_search_output"
    })
    
    # Create and run pipeline
    pipeline = Pipeline(config)
    success = pipeline.run()
    
    if success:
        print("Pipeline completed successfully!")
        
        # Get results
        results = pipeline.get_results()
        
        # Print some information about the results
        if results["search_results"] is not None:
            print("\nSemantic Search Results:")
            print(f"Query: {results['search_results']['query']}")
            print("\nTop Results:")
            for i, (score, message) in enumerate(results['search_results']['results'][:5], 1):
                print(f"{i}. Score: {score:.4f} | {message[:100]}...")
    else:
        print("Pipeline failed!")


def convert_kg_push():
    """Convert ELS_KG_push.ipynb to use the pipeline."""
    print("Converting ELS_KG_push.ipynb to use the Elastic Data Science Pipeline")
    
    # Check if output.csv exists
    if not os.path.exists("output.csv"):
        print("Error: output.csv not found. Run convert_els_pull() first.")
        return
    
    # Create configuration for local data source
    config = Config({
        "data_source": {
            "type": "local",
            "local": {
                "file_path": "output.csv"
            }
        },
        "processing": {
            "normalize": False,  # Don't normalize for knowledge graph
            "output_dir": "./kg_push_output"
        },
        "knowledge_graph": {
            "enabled": True,
            "neo4j_uri": "neo4j://192.168.2.2:7687",  # Remote Neo4j IP from ELS_KG_push.ipynb
            "database": "neo4j",
            "use_llm": True,
            "openai_model": "gpt-4o"  # From ELS_KG_push.ipynb
        },
        "semantic_search": {
            "enabled": False  # Disable semantic search for this example
        },
        "output_dir": "./kg_push_output"
    })
    
    # Create and run pipeline
    pipeline = Pipeline(config)
    success = pipeline.run()
    
    if success:
        print("Pipeline completed successfully!")
        print("Knowledge graph created in Neo4j")
    else:
        print("Pipeline failed!")


def main():
    """Run all conversions."""
    print("Converting notebooks to use the Elastic Data Science Pipeline")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("./els_pull_output", exist_ok=True)
    os.makedirs("./els_norm_output", exist_ok=True)
    os.makedirs("./semantic_search_output", exist_ok=True)
    os.makedirs("./kg_push_output", exist_ok=True)
    
    # Convert each notebook
    convert_els_pull()
    print("\n" + "=" * 50 + "\n")
    
    convert_els_norm()
    print("\n" + "=" * 50 + "\n")
    
    convert_semantic_search()
    print("\n" + "=" * 50 + "\n")
    
    convert_kg_push()
    print("\n" + "=" * 50 + "\n")
    
    print("All conversions completed!")


if __name__ == "__main__":
    main()