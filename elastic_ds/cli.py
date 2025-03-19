"""
Command-line interface for the Elastic Data Science Pipeline.
"""

import argparse
import os
import sys
from typing import Dict, Any, Optional

from .config import Config
from .pipeline import Pipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Elastic Data Science Pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data-source", 
        type=str, 
        choices=["elasticsearch", "api", "local"],
        help="Data source type"
    )
    
    parser.add_argument(
        "--elasticsearch-host", 
        type=str, 
        help="Elasticsearch host URL"
    )
    
    parser.add_argument(
        "--api-url", 
        type=str, 
        help="API base URL"
    )
    
    parser.add_argument(
        "--local-file", 
        type=str, 
        help="Path to local file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output",
        help="Output directory"
    )
    
    parser.add_argument(
        "--normalize", 
        action="store_true",
        help="Enable data normalization"
    )
    
    parser.add_argument(
        "--no-normalize", 
        action="store_false", 
        dest="normalize",
        help="Disable data normalization"
    )
    
    parser.add_argument(
        "--kg-enabled", 
        action="store_true",
        help="Enable knowledge graph creation"
    )
    
    parser.add_argument(
        "--no-kg", 
        action="store_false", 
        dest="kg_enabled",
        help="Disable knowledge graph creation"
    )
    
    parser.add_argument(
        "--neo4j-uri", 
        type=str, 
        help="Neo4j URI"
    )
    
    parser.add_argument(
        "--search-enabled", 
        action="store_true",
        help="Enable semantic search"
    )
    
    parser.add_argument(
        "--no-search", 
        action="store_false", 
        dest="search_enabled",
        help="Disable semantic search"
    )
    
    parser.add_argument(
        "--search-query", 
        type=str, 
        help="Semantic search query"
    )
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key"
    )
    
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        help="Use existing data instead of pulling new data"
    )
    
    parser.add_argument(
        "--pull-new-data",
        action="store_false",
        dest="use_existing_data",
        help="Pull new data (default)"
    )
    
    return parser.parse_args()


def args_to_config(args) -> Dict[str, Any]:
    """Convert command-line arguments to configuration dictionary."""
    config = {}
    
    # Data source configuration
    if args.data_source:
        config["data_source"] = {"type": args.data_source}
        
        if args.data_source == "elasticsearch" and args.elasticsearch_host:
            config["data_source"]["elasticsearch"] = {"host": args.elasticsearch_host}
        
        elif args.data_source == "api" and args.api_url:
            config["data_source"]["api"] = {"base_url": args.api_url}
        
        elif args.data_source == "local" and args.local_file:
            config["data_source"]["local"] = {"file_path": args.local_file}
    
    # Processing configuration
    config["processing"] = {"normalize": args.normalize}
    
    # Output directory
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Knowledge graph configuration
    config["knowledge_graph"] = {"enabled": args.kg_enabled}
    if args.neo4j_uri:
        config["knowledge_graph"]["neo4j_uri"] = args.neo4j_uri
    
    # Semantic search configuration
    config["semantic_search"] = {"enabled": args.search_enabled}
    if args.search_query:
        config["semantic_search"]["default_query"] = args.search_query
    
    # OpenAI API key
    if args.openai_api_key:
        config["knowledge_graph"]["openai_api_key"] = args.openai_api_key
    
    # Data acquisition options
    config["data_acquisition"] = {"use_existing_data": args.use_existing_data}
    
    return config


def main():
    """Main entry point for the CLI."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create configuration
    if args.config:
        config = Config(config_file=args.config)
    else:
        config_dict = args_to_config(args)
        config = Config(config_dict=config_dict)
    
    # Create and run pipeline
    pipeline = Pipeline(config)
    success = pipeline.run()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()