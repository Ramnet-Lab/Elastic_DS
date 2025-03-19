# Elastic Data Science Pipeline

A comprehensive tool for security log analysis with Elasticsearch and Neo4j.

## Overview

This pipeline integrates several data processing components:

1. **Data Acquisition**: Fetch data from Elasticsearch, API, or local files
2. **Data Processing**: Clean and normalize data for analysis and visualization
3. **Knowledge Graph**: Build a Neo4j knowledge graph with LLM-assisted modeling
4. **Semantic Search**: Perform semantic analysis on message content

## Remote Services Configuration

This pipeline is designed to work with remote services. The default configuration uses the following remote IP addresses:

- **Elasticsearch**: http://192.168.2.46:9200
- **API Server**: http://192.168.2.46:5000/files
- **Neo4j**: neo4j://192.168.2.2:7687

These IP addresses are extracted from the original notebooks and should be updated to match your environment if needed.

## Installation

### Prerequisites

- Python 3.8 or higher
- Elasticsearch (remote service at http://192.168.2.46:9200)
- Neo4j (remote service at neo4j://192.168.2.2:7687)
- OpenAI API key (optional, for LLM-assisted modeling)

### Install from source

```bash
git clone https://github.com/yourusername/elastic_ds.git
cd elastic_ds
pip install -e .
```

For Neo4j support:

```bash
pip install -e ".[neo4j]"
```

## Usage

### Command Line Interface

```bash
# Basic usage with Elasticsearch data source
elastic-ds --data-source elasticsearch --elasticsearch-host http://192.168.2.46:9200

# Using a local file
elastic-ds --data-source local --local-file path/to/data.csv

# Using an API
elastic-ds --data-source api --api-url http://192.168.2.46:5000/files

# Disable knowledge graph creation
elastic-ds --data-source local --local-file path/to/data.csv --no-kg

# Specify a semantic search query
elastic-ds --data-source local --local-file path/to/data.csv --search-query "login failure"

# Specify output directory
elastic-ds --data-source local --local-file path/to/data.csv --output-dir ./results

# Use existing data instead of pulling new data
elastic-ds --data-source local --local-file path/to/data.csv --use-existing-data

# Force pulling new data (default behavior)
elastic-ds --data-source local --local-file path/to/data.csv --pull-new-data
```

### Python API

```python
from elastic_ds.config import Config
from elastic_ds.pipeline import Pipeline

# Create configuration
config = Config({
    "data_source": {
        "type": "elasticsearch",
        "elasticsearch": {
            "host": "http://192.168.2.46:9200",
            "index_pattern": "*"
        }
    },
    "processing": {
        "normalize": True,
        "visualization_ready": True
    },
    "knowledge_graph": {
        "enabled": True,
        "neo4j_uri": "neo4j://192.168.2.2:7687",
        "use_llm": True
    },
    "semantic_search": {
        "enabled": True,
        "default_query": "login authentication failure"
    },
    "data_acquisition": {
        "use_existing_data": False  # Set to True to use existing data
    }
})

# Create and run pipeline
pipeline = Pipeline(config)
pipeline.run()

# Get results
results = pipeline.get_results()
```

## Configuration

The pipeline can be configured using a JSON configuration file:

```json
{
  "data_source": {
    "type": "elasticsearch",
    "elasticsearch": {
      "host": "http://192.168.2.46:9200",
      "index_pattern": "*",
      "basic_auth": null,
      "verify_certs": false,
      "timeout": 30,
      "max_retries": 3,
      "retry_on_timeout": true
    }
  },
  "processing": {
    "normalize": true,
    "output_dir": "./processed_data",
    "visualization_ready": true
  },
  "knowledge_graph": {
    "enabled": true,
    "neo4j_uri": "neo4j://192.168.2.2:7687",
    "database": "neo4j",
    "use_llm": true,
    "openai_model": "gpt-4o",
    "openai_api_key": null
  },
  "semantic_search": {
    "enabled": true,
    "model": "all-MiniLM-L6-v2",
    "default_query": "login authentication failure",
    "top_n": 10
  },
  "data_acquisition": {
    "use_existing_data": false
  }
}
```

## Components

### Data Acquisition

- **Elasticsearch**: Connect to Elasticsearch and extract data using the scroll API
- **API**: Connect to a REST API to download CSV files
- **Local**: Load data from local files (CSV, Excel, JSON, Parquet)
- **Existing Data**: Option to use previously acquired data instead of pulling new data

### Data Processing

- **Normalization**: Normalize data for machine learning and visualization
- **Cleaning**: Clean data by handling missing values, converting types, etc.

### Knowledge Graph

- **Modeling**: Create a graph model using Neo4j Runway and LLMs
- **Ingestion**: Ingest data into Neo4j
- **Visualization**: Visualize the graph model

### Semantic Search

- **Embeddings**: Compute embeddings for messages using Sentence Transformers
- **Search**: Perform semantic search using cosine similarity

## Converting Existing Notebooks

The `convert_notebook.py` script demonstrates how to convert the original notebooks to use the new pipeline:

```bash
python convert_notebook.py
```

This will:
1. Pull data from Elasticsearch (ELS_pull.ipynb)
2. Normalize the data (ELS_norm.ipynb)
3. Perform semantic search (Sematic_search.ipynb)
4. Create a knowledge graph (ELS_KG_push.ipynb)

## License

MIT