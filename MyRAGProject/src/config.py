# config.py
# Configuration settings for the RAG application

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file found in the current working directory or parent directories.

# --- Project Root ---
# It's often useful to define the project root for easier path management.
# This assumes config.py is in MyRAGProject/src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# Construct paths relative to PROJECT_ROOT to make them more robust.
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", os.path.join(PROJECT_ROOT, "data/raw/"))
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", os.path.join(PROJECT_ROOT, "data/processed/"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", os.path.join(PROJECT_ROOT, "models/vector_db.faiss"))

# --- Model Configurations ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "distilgpt2") # Using a smaller model for local LLM
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Search Parameters ---
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

# --- API Keys (if applicable, loaded from .env) ---
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

def print_config():
    """Prints the current configuration."""
    print("Configuration loaded:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Raw Data Directory: {RAW_DATA_DIR}")
    print(f"  Processed Data Directory: {PROCESSED_DATA_DIR}")
    print(f"  Vector DB Path: {VECTOR_DB_PATH}")
    print(f"  LLM Model Name: {LLM_MODEL_NAME}")
    print(f"  Embedding Model Name: {EMBEDDING_MODEL_NAME}")
    print(f"  Top K Results: {TOP_K_RESULTS}")

if __name__ == "__main__":
    # This allows you to run python src/config.py to check paths,
    # but ensure .env is in MyRAGProject if running from within MyRAGProject/src
    # or MyRAGProject if running from MyRAGProject
    print_config()
