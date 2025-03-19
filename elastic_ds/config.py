"""
Configuration management for the Elastic Data Science Pipeline.
"""

import os
import json
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the Elastic DS Pipeline."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """
        Initialize configuration from a dictionary or file.

        Args:
            config_dict: Configuration dictionary
            config_file: Path to configuration file (JSON)
        """
        self.config = {
            "data_source": {
                "type": "elasticsearch",  # elasticsearch, api, local
                "elasticsearch": {
                    "host": "http://192.168.2.46:9200",  # Updated to use the remote IP from ELS_pull.ipynb
                    "index_pattern": "*",
                    "basic_auth": None,  # (username, password)
                    "verify_certs": False,
                    "timeout": 30,
                    "max_retries": 3,
                    "retry_on_timeout": True
                },
                "api": {
                    "base_url": "http://192.168.2.46:5000/files",  # Updated to use the same IP as Elasticsearch
                    "save_dir": "csv_downloads"
                },
                "local": {
                    "file_path": "output.csv"
                }
            },
            "processing": {
                "normalize": True,
                "output_dir": "./processed_data",
                "visualization_ready": True  # Enable additional processing for visualization
            },
            "knowledge_graph": {
                "enabled": True,
                "neo4j_uri": "neo4j://192.168.2.2:7687",  # Updated to use the remote IP from ELS_KG_push.ipynb
                "database": "neo4j",
                "use_llm": True,
                "openai_model": "gpt-4o",
                "openai_api_key": None  # Will be loaded from environment
            },
            "semantic_search": {
                "enabled": True,
                "model": "all-MiniLM-L6-v2",
                "default_query": "login authentication failure",
                "top_n": 10
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

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Update a nested dictionary with another dictionary.

        Args:
            d: Target dictionary
            u: Source dictionary

        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Dot-separated key path (e.g., "data_source.type")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Dot-separated key path (e.g., "data_source.type")
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, file_path: str) -> None:
        """
        Save configuration to a file.

        Args:
            file_path: Path to save the configuration
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def __str__(self) -> str:
        """String representation of the configuration."""
        return json.dumps(self.config, indent=2)