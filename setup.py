from setuptools import setup, find_packages

setup(
name="ragbot",
version="0.1.0",
description="Production-ready RAG bot using Pinecone, OpenAI embeddings, and Groq LLM",
author="Nishant Gupta",
package_dir={"": "src"},
packages=find_packages(where="src"),
python_requires=">=3.11",
install_requires=[],
)
