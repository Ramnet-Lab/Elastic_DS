"""
Example script for using the Elastic Data Science Pipeline.
"""

import os
from dotenv import load_dotenv
from elastic_ds.config import Config
from elastic_ds.pipeline import Pipeline


def main():
    """Run an example pipeline."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
    
    # Ask user if they want to use existing data
    use_existing = input("Do you want to use existing data? (y/n): ").lower().strip() == 'y'
    
    # Create configuration
    config = Config({
        "data_source": {
            "type": "multiple",  # Use multiple data sources
            "sources": [
                {
                    "type": "elasticsearch",
                    "config": {
                        "host": "http://192.168.2.46:9200",  # Remote Elasticsearch IP from ELS_pull.ipynb
                        "index_pattern": "*",
                        "verify_certs": False,
                        "timeout": 30,
                        "max_retries": 3,
                        "retry_on_timeout": True,
                        "size": 0  # No limit - retrieve all documents
                    }
                },
                {
                    "type": "api",
                    "config": {
                        "base_url": "http://192.168.2.46:5000/files",  # Zeek data API from zeek_pull.ipynb
                        "save_dir": "./zeek_data"
                    }
                }
            ]
        },
        "processing": {
            "normalize": True,
            "output_dir": "./processed_data",
            "visualization_ready": True
        },
        "knowledge_graph": {
            "enabled": True,  # Enable knowledge graph
            "neo4j_uri": "neo4j://192.168.2.2:7687",  # Remote Neo4j IP from ELS_KG_push.ipynb
            "database": "neo4j",
            "use_llm": True,
            "openai_model": "gpt-4o",
            "openai_api_key": api_key,  # Set API key from environment
            "ingest_to_neo4j": True  # Enable ingestion to Neo4j
        },
        "semantic_search": {
            "enabled": True,
            "model": "all-MiniLM-L6-v2",
            "default_query": "login authentication failure",
            "top_n": 5
        },
        "output_dir": "./example_output",
        "data_acquisition": {
            "use_existing_data": use_existing
        }
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
        
        if results["processed_data"] is not None:
            print(f"Processed data shape: {results['processed_data'].shape}")
        
        if results["graph_model"] is not None:
            print("\nKnowledge Graph Model:")
            print(f"Model created successfully with {len(results['graph_model'].nodes)} nodes")
        
        if results["search_results"] is not None:
            print("\nSemantic Search Results:")
            print(f"Query: {results['search_results']['query']}")
            print("\nTop Results:")
            for i, (score, message) in enumerate(results['search_results']['results'][:3], 1):
                print(f"{i}. Score: {score:.4f} | {message[:100]}...")
    else:
        print("Pipeline failed!")


if __name__ == "__main__":
    main()