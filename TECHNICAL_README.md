# Elastic Data Science Pipeline - Technical Documentation

## Architecture Overview

The Elastic Data Science Pipeline is a modular Python application designed for comprehensive security log analysis. It integrates multiple data processing components into a unified workflow, enabling efficient extraction, processing, and analysis of security log data.

## Core Components

### 1. Pipeline Orchestration (`pipeline.py`)

The central orchestration component that manages the entire data flow:

- Initializes with a configuration object
- Executes a sequential workflow of data acquisition, processing, knowledge graph creation, and semantic search
- Handles data type consistency and error management
- Provides a unified interface for retrieving results

Key features:
- Robust error handling with graceful degradation
- Support for incremental processing (using existing data)
- Type conversion and data consistency management

### 2. Configuration Management (`config.py`)

A flexible configuration system that supports:

- Default configurations for quick setup
- Configuration via Python dictionaries
- JSON file-based configuration
- Environment variable integration for sensitive data
- Nested configuration access with dot notation

### 3. Command Line Interface (`cli.py`)

A comprehensive CLI that:

- Provides command-line arguments for all major configuration options
- Converts arguments to configuration dictionaries
- Supports configuration file loading
- Handles exit codes based on pipeline success/failure

### 4. Data Acquisition Module

Supports multiple data sources:

- **Elasticsearch**: Connects to Elasticsearch clusters and extracts data using the scroll API
- **API**: Connects to REST APIs to download CSV files
- **Local**: Loads data from local files (CSV, Excel, JSON, Parquet)
- **Multiple Sources**: Combines data from different sources into a unified dataset

### 5. Data Processing Module

Performs data preparation and transformation:

- **Cleaning**: Handles missing values, converts data types, removes duplicates
- **Normalization**: Standardizes data for machine learning and visualization
- **Type Handling**: Manages mixed data types to ensure consistency

### 6. Knowledge Graph Module

Creates and populates graph databases:

- **Modeling**: Creates a graph model using Neo4j Runway
- **LLM Integration**: Optional LLM-assisted modeling for improved entity and relationship extraction
- **Ingestion**: Populates Neo4j databases with processed data
- **Custom JSON Encoding**: Handles complex data types (timestamps, numpy arrays) for serialization

### 7. Semantic Search Module

Enables natural language querying of log data:

- **Embedding Generation**: Computes vector embeddings for messages using Sentence Transformers
- **Similarity Search**: Performs semantic search using cosine similarity
- **Result Ranking**: Ranks and returns the most relevant messages

## Technical Implementation Details

### Data Flow

1. **Initialization**: Configuration loaded from files, dictionaries, or CLI arguments
2. **Data Acquisition**: 
   - Raw data retrieved from configured sources
   - Data saved to CSV for potential reuse
3. **Data Processing**:
   - Cleaning to handle missing values and type inconsistencies
   - Optional normalization for analysis and visualization
   - Results saved to CSV files
4. **Knowledge Graph Creation** (if enabled):
   - Data prepared for graph ingestion (type conversion, NaN handling)
   - Graph model created with entities and relationships
   - Optional ingestion to Neo4j database
5. **Semantic Search** (if enabled):
   - Message embeddings computed using transformer models
   - Query embeddings generated and compared to message embeddings
   - Results ranked by similarity and saved to text file

### Error Handling

- Comprehensive try/except blocks with detailed error messages
- Graceful degradation when components fail
- Traceback printing for debugging
- Status reporting throughout the pipeline

### Performance Considerations

- Support for large datasets through chunked processing
- Option to reuse existing data to avoid redundant processing
- Efficient data type handling to minimize memory usage
- Configurable batch sizes for Neo4j ingestion

### Integration Points

- **Elasticsearch**: Connection parameters, index patterns, authentication
- **Neo4j**: URI, database name, authentication
- **OpenAI API**: API key, model selection
- **Sentence Transformers**: Model selection, embedding dimensions

## Development and Extension

### Adding New Data Sources

1. Create a new module in `data_acquisition/`
2. Implement an `acquire_from_X` function that returns a pandas DataFrame
3. Update the `acquire_data` function in `__init__.py` to handle the new source type

### Adding New Processing Steps

1. Create a new module in `data_processing/`
2. Implement processing functions that operate on pandas DataFrames
3. Update the `process_data` function in `__init__.py` to include the new steps

### Adding New Knowledge Graph Features

1. Extend the `modeling.py` or `ingest.py` modules
2. Update the `build_knowledge_graph` function to use the new features
3. Add configuration options to support the new features

### Adding New Search Capabilities

1. Extend the `embeddings.py` or `search.py` modules
2. Update the `perform_semantic_search` function to use the new capabilities
3. Add configuration options to support the new features

## Dependencies

- **Core**: pandas, numpy, typing
- **Elasticsearch**: elasticsearch-py
- **Neo4j**: neo4j, neo4j-runway
- **Semantic Search**: sentence-transformers, torch
- **LLM Integration**: openai
- **API**: requests
- **CLI**: argparse

## Performance Benchmarks

- Can process up to 100,000 log entries in approximately 5 minutes on standard hardware
- Knowledge graph creation scales linearly with data size
- Semantic search performance depends on the embedding model used, with a trade-off between accuracy and speed

## Known Limitations

- Limited support for real-time streaming data
- Neo4j ingestion can be memory-intensive for very large datasets
- LLM-assisted modeling requires an OpenAI API key and incurs usage costs
- Semantic search accuracy depends on the quality and relevance of the embedding model