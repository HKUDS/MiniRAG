# utils.py
# Utility functions for the RAG application

import os
import dotenv

def load_env_vars():
    """Loads environment variables from .env file."""
    dotenv.load_dotenv()
    # Example: api_key = os.getenv("API_KEY")
    print("Environment variables loaded.")

def some_helper_function():
    """A placeholder for a utility function."""
    print("Helper function called.")
    return True

# TODO: Add more utility functions as needed (e.g., text cleaning, file I/O helpers)
