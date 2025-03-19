"""
Elasticsearch data acquisition module.
"""

import pandas as pd
from typing import Dict, Any, Optional
from elasticsearch import Elasticsearch


def acquire_from_elasticsearch(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Acquire data from Elasticsearch.

    Args:
        config: Elasticsearch configuration

    Returns:
        DataFrame with Elasticsearch data if successful, None otherwise
    """
    try:
        # Get Elasticsearch connection details
        host = config.get("host", "http://localhost:9200")
        index_pattern = config.get("index_pattern", "*")
        verify_certs = config.get("verify_certs", True)
        timeout = config.get("timeout", 30)
        max_retries = config.get("max_retries", 3)
        retry_on_timeout = config.get("retry_on_timeout", True)
        
        # Get size limit (0 means no limit)
        size = config.get("size", 0)
        
        # Connect to Elasticsearch
        es = Elasticsearch(
            [host],
            verify_certs=verify_certs,
            request_timeout=timeout,
            max_retries=max_retries,
            retry_on_timeout=retry_on_timeout
        )
        
        # Query Elasticsearch
        query_body = {
            "query": {"match_all": {}},
            "sort": [{"@timestamp": {"order": "desc"}}]
        }
        
        # Add size if specified and not 0
        if size > 0:
            query_body["size"] = size
        
        scroll_duration = "2m"  # Duration for which the scroll context is maintained
        
        # Execute initial search
        page = es.search(index=index_pattern, body=query_body, scroll=scroll_duration)
        scroll_id = page.get('_scroll_id')
        hits = page['hits']['hits']
        total_hits = page['hits']['total']['value'] if 'total' in page['hits'] else None
        
        if total_hits:
            print(f"Found {total_hits} documents in Elasticsearch")
        
        # Collect all documents using the scroll API
        documents = []
        documents.extend(hits)
        
        # Continue scrolling until no more hits
        while True:
            page = es.scroll(scroll_id=scroll_id, scroll=scroll_duration)
            hits = page['hits']['hits']
            
            if not hits:
                break
                
            documents.extend(hits)
            print(f"Retrieved {len(documents)} documents so far...")
            
            # If size is specified and we've reached it, stop
            if size > 0 and len(documents) >= size:
                documents = documents[:size]
                break
                
            scroll_id = page.get('_scroll_id')
        
        # Build DataFrame from documents
        if documents:
            data = []
            for doc in documents:
                row = {}
                row['_index'] = doc.get('_index', '')
                source = doc.get('_source', {})
                row.update(source)
                data.append(row)
                
            df = pd.DataFrame(data)
            
            # Parse and flatten JSON columns
            json_columns = ['agent', 'ecs', 'host', 'input', 'log']
            
            for col in json_columns:
                if col in df.columns:
                    # Parse JSON-like strings
                    df[col] = df[col].apply(parse_json_value)
                    parsed = df[col].apply(lambda x: x if isinstance(x, (dict, list)) else {})
                    
                    # If the column contains only dictionaries, flatten it
                    if parsed.apply(lambda x: isinstance(x, dict)).all():
                        expanded = pd.json_normalize(parsed)
                        expanded.columns = [f"{col}.{subcol}" for subcol in expanded.columns]
                        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
                    else:
                        # Leave the column as is
                        df[col] = parsed
            
            # Convert @timestamp to datetime
            if '@timestamp' in df.columns:
                df['@timestamp'] = pd.to_datetime(df['@timestamp'], errors='coerce')
                
            return df
        else:
            print("No documents found in Elasticsearch")
            return None
            
    except Exception as e:
        print(f"Error acquiring data from Elasticsearch: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_json_value(val: Any) -> Any:
    """
    Convert a JSON-like string into a dictionary.
    
    Args:
        val: Value to parse
        
    Returns:
        Parsed value if successful, original value otherwise
    """
    import ast
    
    if isinstance(val, str):
        val = val.strip()
        if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
            try:
                return ast.literal_eval(val)
            except Exception:
                return val  # Return the original string if parsing fails
    return val