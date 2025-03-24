![Elastic Data Science Pipeline](example_images/splashimg.webp)

# Elastic Data Science Pipeline

A comprehensive security log analysis solution that transforms raw security data into actionable intelligence through data acquisition, processing, knowledge graph modeling, and semantic search capabilities.

## Executive Summary

The Elastic Data Science Pipeline empowers security teams to quickly identify patterns, detect anomalies, and respond to threats more effectively. By integrating multiple data sources and advanced analysis techniques, this tool provides a unified approach to security log analysis.

**Key Benefits:**
- **Unified Analysis:** Consolidates multiple security data sources into a single, coherent pipeline
- **Knowledge Graph Visualization:** Transforms complex security data into intuitive graph visualizations
- **AI-Powered Semantic Search:** Enables natural language querying of security logs
- **LLM-Enhanced Analysis:** Leverages Large Language Models to improve entity extraction
- **Accelerated Incident Response:** Reduces mean time to detect (MTTD) and respond (MTTR)
- **Operational Efficiency:** Automates repetitive data processing and analysis tasks

## Installation

### Prerequisites

- Python 3.8 or higher
- Elasticsearch (remote service at http://192.168.2.46:9200 by default)
- Neo4j (remote service at neo4j://192.168.2.2:7687 by default)
- OpenAI API key (optional, for LLM-assisted modeling)

### Install from Source

```bash
git clone https://github.com/yourusername/elastic_ds.git
cd elastic_ds
pip install -e .
```

For Neo4j support:

```bash
pip install -e ".[neo4j]"
```

### Dependencies

The project requires the following Python packages:
- **elasticsearch>=7.0.0**: For connecting to and querying Elasticsearch
- **pandas>=1.0.0**: For data manipulation and analysis
- **numpy>=1.18.0**: For numerical operations
- **matplotlib>=3.1.0**: For data visualization
- **networkx>=2.4**: For graph operations
- **sentence-transformers>=2.0.0**: For computing text embeddings
- **tqdm>=4.45.0**: For progress bars
- **pyyaml>=5.1**: For YAML file parsing
- **openai>=1.0.0**: For LLM integration
- **neo4j-runway>=0.1.0**: (Optional) For Neo4j knowledge graph creation. See [Neo4j Runway GitHub repository](https://github.com/a-s-g93/neo4j-runway) for more information.

## Usage

### Command Line Interface

The pipeline provides a comprehensive command-line interface with numerous options:

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

# Provide OpenAI API key for LLM-assisted modeling
elastic-ds --data-source local --local-file path/to/data.csv --openai-api-key YOUR_API_KEY
```

### Python API

The pipeline can be used programmatically in Python scripts:

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

# Access individual components
raw_data = results["raw_data"]  # pandas DataFrame with raw data
processed_data = results["processed_data"]  # pandas DataFrame with processed data
graph_model = results["graph_model"]  # Neo4j Runway graph model
search_results = results["search_results"]  # Semantic search results
```

## Architecture and Components

The pipeline integrates several modular components into a unified workflow:

### 1. Pipeline Orchestration (`pipeline.py`)

The central component that manages the entire data flow:
- Initializes with a configuration object
- Executes a sequential workflow of data acquisition, processing, knowledge graph creation, and semantic search
- Handles data type consistency with automatic type conversion
- Provides robust error handling with graceful degradation
- Supports incremental processing using existing data
- Implements a unified interface for retrieving results

Implementation details:
```python
def run(self) -> bool:
    """Run the pipeline."""
    try:
        # Step 1: Acquire data or use existing data
        print("\n=== Step 1: Data Acquisition ===")
        
        # Check if we should use existing data
        use_existing_data = self.config.get("data_acquisition", {}).get("use_existing_data", False)
        raw_csv = os.path.join(self.output_dir, "raw_data.csv")
        
        if use_existing_data and os.path.exists(raw_csv):
            print("Using existing data from previous run...")
            self.df = pd.read_csv(raw_csv, low_memory=False)
            self.df = handle_mixed_types(self.df)
        else:
            print("Acquiring new data...")
            self.df = acquire_data(self.config.get("data_source", {}))
            self.df = handle_mixed_types(self.df)
            self.df.to_csv(raw_csv, index=False, encoding="utf-8")
        
        # Step 2: Process data
        print("\n=== Step 2: Processing Data ===")
        self.df_processed, self.mapping_dict = process_data(self.df, self.config.get("processing", {}))
        
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
            # Build knowledge graph
            self.graph_model = build_knowledge_graph(df_for_kg, self.config.get("knowledge_graph", {}))
        
        # Step 4: Perform semantic search (if enabled)
        if self.config.get("semantic_search", {}).get("enabled", True):
            print("\n=== Step 4: Performing Semantic Search ===")
            self.search_results = perform_semantic_search(self.df, self.config.get("semantic_search", {}))
        
        return True
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
```

### 2. Configuration Management (`config.py`)

A flexible configuration system that supports:
- Default configurations for quick setup
- Configuration via Python dictionaries
- JSON file-based configuration
- Environment variable integration for sensitive data
- Nested configuration access with dot notation

Implementation details:
```python
class Config:
    """Configuration manager for the Elastic DS Pipeline."""

    def __init__(self, config_dict=None, config_file=None):
        """Initialize configuration from a dictionary or file."""
        self.config = {
            "data_source": {
                "type": "elasticsearch",
                "elasticsearch": {
                    "host": "http://192.168.2.46:9200",
                    "index_pattern": "*",
                    "basic_auth": None,
                    "verify_certs": False,
                    "timeout": 30,
                    "max_retries": 3,
                    "retry_on_timeout": True
                },
                # Additional default configuration...
            }
        }
        
        # Override defaults with provided config
        if config_dict:
            self._update_nested_dict(self.config, config_dict)
            
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._update_nested_dict(self.config, file_config)
                
        # Load API keys from environment
        if not self.config["knowledge_graph"]["openai_api_key"]:
            self.config["knowledge_graph"]["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
```

### 3. Data Acquisition Module

Supports multiple data sources with specialized implementations:

#### Elasticsearch Acquisition (`elasticsearch.py`)
- Connects to Elasticsearch clusters using the official Python client
- Extracts data using the scroll API for efficient retrieval of large datasets
- Handles pagination and size limits
- Parses and flattens nested JSON fields
- Converts timestamps to pandas datetime objects

Implementation details:
```python
def acquire_from_elasticsearch(config):
    """Acquire data from Elasticsearch."""
    # Connect to Elasticsearch
    es = Elasticsearch(
        [config.get("host", "http://localhost:9200")],
        verify_certs=config.get("verify_certs", True),
        request_timeout=config.get("timeout", 30),
        max_retries=config.get("max_retries", 3),
        retry_on_timeout=config.get("retry_on_timeout", True)
    )
    
    # Use scroll API for efficient retrieval
    query_body = {"query": {"match_all": {}}, "sort": [{"@timestamp": {"order": "desc"}}]}
    scroll_duration = "2m"
    page = es.search(index=config.get("index_pattern", "*"), body=query_body, scroll=scroll_duration)
    
    # Collect all documents
    documents = []
    documents.extend(page['hits']['hits'])
    
    # Continue scrolling until no more hits
    while True:
        page = es.scroll(scroll_id=page.get('_scroll_id'), scroll=scroll_duration)
        hits = page['hits']['hits']
        if not hits:
            break
        documents.extend(hits)
        
    # Build DataFrame from documents
    data = []
    for doc in documents:
        row = {'_index': doc.get('_index', '')}
        row.update(doc.get('_source', {}))
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Parse and flatten JSON columns
    json_columns = ['agent', 'ecs', 'host', 'input', 'log']
    for col in json_columns:
        if col in df.columns:
            # Parse JSON-like strings
            df[col] = df[col].apply(parse_json_value)
            # Flatten if column contains dictionaries
            if df[col].apply(lambda x: isinstance(x, dict)).all():
                expanded = pd.json_normalize(df[col])
                expanded.columns = [f"{col}.{subcol}" for subcol in expanded.columns]
                df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
    
    return df
```

#### API Acquisition (`api.py`)
- Connects to REST APIs to download CSV files
- Supports authentication and custom headers
- Handles HTTP errors and retries

#### Local File Acquisition (`local.py`)
- Loads data from local files in various formats (CSV, Excel, JSON, Parquet)
- Handles different file encodings and delimiters
- Supports automatic schema detection

### 4. Data Processing Module

Performs data preparation and transformation:

#### Cleaning (`cleaning.py`)
- Handles missing values with configurable strategies
- Converts data types to ensure consistency
- Removes duplicates and outliers
- Filters data based on configurable criteria

#### Normalization (`normalization.py`)
- Standardizes data for machine learning and visualization
- Scales numerical features
- Encodes categorical variables
- Prepares data for time-series analysis

### 5. Knowledge Graph Module

Creates and populates graph databases using [Neo4j Runway](https://github.com/a-s-g93/neo4j-runway):

#### Modeling (`modeling.py`)
- Creates a graph model using Neo4j Runway
- Implements LLM-assisted modeling using OpenAI's GPT models
- Generates dynamic context information and data dictionary
- Handles complex data types and serialization

Implementation details:
```python
def create_graph_model(df, config):
    """Create a graph model from the DataFrame using Neo4j Runway."""
    # Import Neo4j Runway components
    from neo4j_runway import Discovery, GraphDataModeler, UserInput
    from neo4j_runway.llm.openai import OpenAIDiscoveryLLM, OpenAIDataModelingLLM
    
    # Get API key from configuration or environment
    api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
    
    # Set up OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Generate context information using GPT
    context_info = generate_dynamic_context(df, client, config)
    general_description = context_info["general_description"]
    use_cases = context_info["use_cases"]
    
    # Generate data dictionary using GPT
    data_dictionary = generate_data_dictionary(df, general_description, use_cases, client, config)
    
    # Initialize LLMs for Neo4j Runway
    model_name = config.get("openai_model", "gpt-4o")
    llm_disc = OpenAIDiscoveryLLM(model_name=model_name, model_params={"temperature": 0})
    llm_dm = OpenAIDataModelingLLM(model_name=model_name, model_params={"temperature": 0.5})
    
    # Create UserInput object for Neo4j Runway
    user_input = UserInput(
        file_path="data.csv",
        general_description=general_description,
        use_cases=use_cases,
        data_dictionary=data_dictionary
    )
    
    # Initialize Discovery instance with DataFrame
    disc = Discovery(llm=llm_disc, data=df)
    disc.run()
    
    # Create Graph Data Modeler
    gdm = GraphDataModeler(llm=llm_dm, discovery=disc)
    gdm.create_initial_model()
    
    return gdm.current_model
```

#### Ingestion (`ingest.py`)
- Populates Neo4j databases with processed data
- Implements efficient batch ingestion
- Handles relationship creation and constraints

#### Visualization (`visualization.py`)
- Visualizes the graph model using NetworkX and Matplotlib
- Supports interactive visualization with Pyvis
- Exports graph data for external visualization tools

#### Graph Visualization Example

![Network Traffic Knowledge Graph](example_images/graph.png)

The image above illustrates a knowledge graph visualization of network traffic data generated by the pipeline. This visualization demonstrates:

- **IP Address Relationships**: Source IP addresses (shown as nodes) connecting to destination IP addresses, revealing communication patterns and potential network segments
- **Protocol Analysis**: Different protocols (HTTP, HTTPS, DNS, etc.) used in communications, represented as relationship properties
- **Port Utilization**: Destination ports accessed by source IPs, helping identify services and potential vulnerabilities
- **Data Volume Mapping**: The thickness of connection lines represents data volume, highlighting high-bandwidth connections that may indicate data exfiltration or normal high-volume services
- **Temporal Patterns**: When analyzed over time, the graph can reveal periodic communication patterns or anomalous connections

This type of visualization enables security analysts to quickly identify:
- Unusual communication patterns between hosts
- Potential lateral movement within a network
- Data exfiltration attempts based on volume anomalies
- Service usage across the network infrastructure

The knowledge graph transforms raw log data into an intuitive visual representation that reveals relationships that would be difficult to detect in tabular data formats.

### 6. Semantic Search Module

Enables natural language querying of log data:

#### Embeddings (`embeddings.py`)
- Computes vector embeddings for messages using Sentence Transformers
- Supports different embedding models
- Implements efficient batch processing

Implementation details:
```python
def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Compute embeddings for a list of texts."""
    from sentence_transformers import SentenceTransformer
    
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return np.array(embeddings)

def compute_query_embedding(query, model_name="all-MiniLM-L6-v2"):
    """Compute embedding for a query."""
    from sentence_transformers import SentenceTransformer
    
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Compute embedding
    embedding = model.encode([query])[0]
    
    return np.array(embedding)
```

#### Search (`search.py`)
- Performs semantic search using cosine similarity
- Ranks and returns the most relevant messages
- Supports customizable result counts

Implementation details:
```python
def semantic_search(query, texts, embeddings, top_n=10):
    """Perform semantic search using cosine similarity."""
    # Compute query embedding
    query_embedding = compute_query_embedding(query)
    
    # Compute cosine similarity
    similarities = compute_cosine_similarity(query_embedding, embeddings)
    
    # Get top N results
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Create result list
    results = []
    for idx in top_indices:
        sim_score = similarities[idx]
        text = texts[idx]
        results.append((sim_score, text))
    
    return results

def compute_cosine_similarity(query_embedding, embeddings):
    """Compute cosine similarity between a query embedding and a set of embeddings."""
    # Compute dot products
    dot_products = np.dot(embeddings, query_embedding)
    
    # Compute norms
    message_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    
    # Compute cosine similarities
    cosine_similarities = dot_products / (message_norms * query_norm)
    
    return cosine_similarities
```

#### Semantic Search Example

![Semantic Search Demo](example_images/semantic.gif)

The GIF above demonstrates the semantic search capability of the Elastic Data Science Pipeline. This feature allows security analysts to:

- **Search for Conceptually Related Terms**: The example shows a search for anything related to "login", "failure", or "error" events
- **Natural Language Understanding**: Unlike traditional keyword search, semantic search understands the meaning behind queries
- **Context-Aware Results**: Returns results that are conceptually similar even when they don't contain the exact search terms
- **Relevance Ranking**: Results are automatically ranked by semantic similarity to the query
- **Immediate Insights**: Quickly identifies security events of interest without requiring exact query syntax

This semantic search capability significantly reduces the time needed to find relevant security events in large datasets, enabling faster incident response and more effective threat hunting.

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

## Technical Implementation Details

### Data Flow

1. **Initialization**: 
   - Configuration loaded from files, dictionaries, or CLI arguments
   - Output directories created
   - Pipeline components initialized

2. **Data Acquisition**: 
   - Raw data retrieved from configured sources (Elasticsearch, API, local files)
   - Data saved to CSV for potential reuse
   - Data types handled to ensure consistency

3. **Data Processing**:
   - Cleaning to handle missing values and type inconsistencies
   - Optional normalization for analysis and visualization
   - Results saved to CSV files

4. **Knowledge Graph Creation** (if enabled):
   - Data prepared for graph ingestion (type conversion, NaN handling)
   - Context information and data dictionary generated using LLMs
   - Graph model created with entities and relationships
   - Optional ingestion to Neo4j database

5. **Semantic Search** (if enabled):
   - Message embeddings computed using transformer models
   - Query embeddings generated and compared to message embeddings
   - Results ranked by similarity and saved to text file

### Error Handling

The pipeline implements comprehensive error handling:
- Try/except blocks with detailed error messages
- Graceful degradation when components fail
- Traceback printing for debugging
- Status reporting throughout the pipeline

Example from `pipeline.py`:
```python
try:
    # Pipeline steps...
except Exception as e:
    print(f"Error in pipeline: {e}")
    import traceback
    traceback.print_exc()
    return False
```

### Performance Considerations

- Support for large datasets through chunked processing
- Option to reuse existing data to avoid redundant processing
- Efficient data type handling to minimize memory usage
- Configurable batch sizes for Neo4j ingestion
- Can process up to 100,000 log entries in approximately 5 minutes on standard hardware

### Type Handling

The pipeline includes specialized handling for mixed data types:
```python
def handle_mixed_types(df):
    """Handle mixed data types in a DataFrame."""
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
```

## Business Value and ROI

### Cost Reduction
- Reduces security analyst time spent on manual data correlation by up to 1M%
- Decreases incident response costs through faster detection and remediation

### Risk Mitigation
- Improves detection of sophisticated attacks that evade traditional security tools
- Reduces the risk of data breaches through earlier threat detection
- Minimizes potential financial and reputational damage from security incidents

### Operational Improvements
- Enhances security team productivity through workflow automation
- Provides better visibility for executive reporting and compliance documentation

## Development and Extension

### Adding New Data Sources
1. Create a new module in `data_acquisition/`
2. Implement an `acquire_from_X` function that returns a pandas DataFrame
3. Update the `acquire_data` function in `__init__.py` to handle the new source type

Example:
```python
def acquire_from_new_source(config):
    """Acquire data from a new source."""
    # Implementation details...
    return df

# In __init__.py
def acquire_data(config):
    """Acquire data from the configured source."""
    source_type = config.get("type", "elasticsearch")
    
    if source_type == "elasticsearch":
        return acquire_from_elasticsearch(config.get("elasticsearch", {}))
    elif source_type == "api":
        return acquire_from_api(config.get("api", {}))
    elif source_type == "local":
        return acquire_from_local(config.get("local", {}))
    elif source_type == "new_source":  # Add new source type
        return acquire_from_new_source(config.get("new_source", {}))
    else:
        print(f"Error: Unknown data source type: {source_type}")
        return None
```

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

## Converting Existing Notebooks

The `convert_notebook.py` script demonstrates how to convert original notebooks to use the new pipeline:

```bash
python convert_notebook.py
```

This will:
1. Pull data from Elasticsearch (ELS_pull.ipynb)
2. Normalize the data (ELS_norm.ipynb)
3. Perform semantic search (Sematic_search.ipynb)
4. Create a knowledge graph (ELS_KG_push.ipynb)

## Known Limitations

- Limited support for real-time streaming data
- Neo4j ingestion can be memory-intensive for very large datasets
- LLM-assisted modeling requires an OpenAI API key and may incur usage costs. It can also be run with local LLMs, though with the typical limitations therein.
- Semantic search accuracy depends on the quality and relevance of the embedding model, which is also dependent on hardware resources.
- The pipeline is designed for batch processing rather than real-time analysis
- Knowledge graph visualization requires additional tools for interactive exploration

## License

MIT
