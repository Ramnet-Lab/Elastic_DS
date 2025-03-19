"""
Graph modeling module for the Elastic Data Science Pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional, List
from openai import OpenAI
from datetime import datetime


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Pandas and NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def create_graph_model(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Any]:
    """
    Create a graph model from the DataFrame using Neo4j Runway.

    Args:
        df: Input DataFrame
        config: Knowledge graph configuration

    Returns:
        Graph model if successful, None otherwise
    """
    try:
        # Import Neo4j Runway components
        from neo4j_runway import Discovery, GraphDataModeler, UserInput
        from neo4j_runway.llm.openai import OpenAIDiscoveryLLM, OpenAIDataModelingLLM
    except ImportError:
        print("Neo4j Runway is not installed. Please install it with: pip install neo4j-runway")
        return None

    # Get API key from configuration or environment
    api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        print("OpenAI API key not found in configuration or environment")
        return None

    # Ensure API key is set in environment for Neo4j Runway
    os.environ["OPENAI_API_KEY"] = api_key

    # Set up OpenAI client
    client = OpenAI(api_key=api_key)

    # Generate context information using GPT
    context_info = generate_dynamic_context(df, client, config)
    if not context_info:
        print("Failed to generate context information")
        return None

    general_description = context_info["general_description"]
    use_cases = context_info["use_cases"]

    # Generate data dictionary using GPT
    data_dictionary = generate_data_dictionary(df, general_description, use_cases, client, config)
    if not data_dictionary:
        print("Failed to generate data dictionary")
        return None

    # Initialize LLMs for Neo4j Runway
    model_name = config.get("openai_model", "gpt-4o")
    
    llm_disc = OpenAIDiscoveryLLM(
        model_name=model_name,
        model_params={"temperature": 0}
    )

    llm_dm = OpenAIDataModelingLLM(
        model_name=model_name,
        model_params={"temperature": 0.5}
    )

    # Create UserInput object for Neo4j Runway
    user_input = UserInput(
        file_path="data.csv",
        general_description=general_description,
        use_cases=use_cases,
        data_dictionary=data_dictionary
    )

    # Initialize Discovery instance with DataFrame
    disc = Discovery(llm=llm_disc, data=df)

    # Run Discovery
    disc.run()

    # Create Graph Data Modeler
    gdm = GraphDataModeler(llm=llm_dm, discovery=disc)

    # Create initial model
    gdm.create_initial_model()

    return gdm.current_model


def generate_dynamic_context(df: pd.DataFrame, client: OpenAI, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate dynamic context information using GPT.

    Args:
        df: Input DataFrame
        client: OpenAI client
        config: Knowledge graph configuration

    Returns:
        Dictionary with general_description and use_cases
    """
    columns_list = list(df.columns)
    
    # Convert DataFrame to dict and handle Timestamp objects
    sample_data = df.head(3).to_dict(orient="records")
    
    # Convert sample data to JSON-serializable format
    sample_data_serializable = json_serialize_dataframe_records(sample_data)

    # Use custom JSON encoder to handle Timestamp objects
    sample_data_json = json.dumps(sample_data_serializable, indent=2, cls=CustomJSONEncoder)

    prompt = f"""I have a CSV file with the following column headers:

{columns_list}

Here is a small sample of the data:

{sample_data_json}


Analyze this data and generate:
1. A general description of what this dataset represents.
2. A list of 5 potential use cases relevant to analyzing this data.
3. general description and usecases should be targeted at scoping an llm to look at potential assosiations in data.


Output a JSON object with two keys: "general_description" (a single-line string) and "use_cases" (a list of short bullet points).
Output only a complete valid JSON object (including the curly braces) without any markdown formatting or code fences.
"""

    model_name = config.get("openai_model", "gpt-4o")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.1
        )
        
        context_str = response.choices[0].message.content.strip()
        return extract_json(context_str)
    except Exception as e:
        print(f"Error generating dynamic context: {e}")
        return None


def generate_data_dictionary(df: pd.DataFrame, general_description: str, use_cases: List[str], 
                            client: OpenAI, config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Generate a data dictionary using GPT.

    Args:
        df: Input DataFrame
        general_description: General description of the data
        use_cases: List of use cases
        client: OpenAI client
        config: Knowledge graph configuration

    Returns:
        Dictionary mapping column names to descriptions
    """
    columns_list = list(df.columns)

    prompt = f"""I have a CSV file with the following column headers:

{columns_list}

The file contains network data.

General description: "{general_description}"

Use cases: {use_cases}

Generate a JSON object that maps each column header to a concise description of what the column represents.

Ensure the descriptions are tailored to this data context and are targeted and stearing an llm to building data assostiations.

Output only a complete valid JSON object (including the curly braces) without any markdown formatting or code fences.
"""

    model_name = config.get("openai_model", "gpt-4o")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.1
        )
        
        data_dict_str = response.choices[0].message.content.strip()
        return extract_json(data_dict_str)
    except Exception as e:
        print(f"Error generating data dictionary: {e}")
        return None



def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and validate JSON from text.

    Args:
        text: Text containing JSON

    Returns:
        Parsed JSON object
    """
    # First, try to extract JSON from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\\s*\\n([\\s\\S]*?)\\n```', text)
    if code_block_match:
        potential_json = code_block_match.group(1).strip()
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            # If parsing fails, continue with other methods
            pass
    
    # Try to extract JSON objects (with curly braces)
    object_match = re.search(r'\\{[\\s\\S]*?\\}', text, re.DOTALL)
    if object_match:
        potential_json = object_match.group()
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError as e:
            # Try to fix common JSON formatting issues
            fixed_json = None
            try:
                # Fix missing quotes around property names
                fixed_json = re.sub(r'(\\{|\\,)\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*:', r'\\1 "\\2":', potential_json)
                
                # Fix single quotes to double quotes
                fixed_json = fixed_json.replace("'", "\"")
                
                # Fix trailing commas
                fixed_json = re.sub(r',\\s*(\\}|\\])', r'\\1', fixed_json)
                
                return json.loads(fixed_json)
            except:
                pass
    
    # Try to extract JSON arrays (with square brackets)
    array_match = re.search(r'\\[.*?\\]', text, re.DOTALL)
    if array_match:
        potential_json = array_match.group()
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            # If parsing fails, continue with other methods
            pass
    
    # If all extraction methods fail, create a fallback JSON
    print("No valid JSON object detected in response")
    
    # Extract general description
    description_match = re.search(r'(?:general description|description)[:\\s]+"?([^\\"\\n]+)"?',
                                 text, re.IGNORECASE)
    general_description = description_match.group(1).strip() if description_match else None
    
    # Extract use cases
    use_cases = []
    use_case_section = re.search(r'(?:use cases|use-cases)[:\\s]+(.+?)(?:\\n\\n|\\Z)',
                               text, re.IGNORECASE | re.DOTALL)
    
    if use_case_section:
        use_case_text = use_case_section.group(1)
        # Extract bullet points or numbered items
        items = re.findall(r'(?:[-*]|\\d+\\.)\\s+([^\\n]+)', use_case_text)
        if items:
            use_cases = [item.strip() for item in items]
        else:
            # If no bullet points, try to split by commas or newlines
            items = re.split(r',|\\n', use_case_text)
            use_cases = [item.strip() for item in items if item.strip()]
    
    # If we have at least a description or use cases, create a fallback JSON
    if general_description or use_cases:
        fallback = {}
        if general_description:
            fallback["general_description"] = general_description
        if use_cases:
            fallback["use_cases"] = use_cases
        return fallback
    
    print("Failed to generate data dictionary")
    return None


def json_serialize_dataframe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert DataFrame records to JSON-serializable format.
    
    Args:
        records: List of DataFrame records
        
    Returns:
        JSON-serializable list of records
    """
    serializable_records = []
    
    for record in records:
        serializable_record = {}
        for key, value in record.items():
            # Handle Pandas Timestamp objects
            if isinstance(value, (pd.Timestamp, datetime)):
                serializable_record[key] = value.isoformat()
            # Handle numpy data types
            elif isinstance(value, np.integer):
                serializable_record[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_record[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_record[key] = value.tolist()
            # Handle pandas Series or DataFrame
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                serializable_record[key] = json_serialize_dataframe_records(value.to_dict(orient="records"))
            # Handle NaN, None, etc.
            elif pd.isna(value):
                serializable_record[key] = None
            # Handle other non-serializable types
            elif isinstance(value, (set, frozenset)):
                serializable_record[key] = list(value)
            else:
                serializable_record[key] = value
                
        serializable_records.append(serializable_record)
    
    return serializable_records