"""
Setup script for the Elastic Data Science Pipeline.
"""

from setuptools import setup, find_packages

setup(
    name="elastic_ds",
    version="0.1.0",
    description="A comprehensive tool for security log analysis with Elasticsearch and Neo4j",
    author="Elastic DS Team",
    packages=find_packages(),
    install_requires=[
        "elasticsearch>=7.0.0",
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "networkx>=2.4",
        "sentence-transformers>=2.0.0",
        "tqdm>=4.45.0",
        "pyyaml>=5.1",
        "openai>=1.0.0",
    ],
    extras_require={
        "neo4j": ["neo4j-runway>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "elastic-ds=elastic_ds.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)