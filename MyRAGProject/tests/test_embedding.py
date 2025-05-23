# tests/test_embedding.py

import pytest
import os
import sys

# Add project root to sys.path to allow importing MyRAGProject
# This assumes tests are run from the 'MyRAGProject' directory or its parent
# A better way might be to install the package in editable mode (pip install -e .)
# or structure the project so that MyRAGProject is directly in PYTHONPATH.
PROJECT_ROOT_FROM_TEST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT_FROM_TEST)

# Now we can import from MyRAGProject.src
try:
    from src.core import EmbeddingModel
    from src.config import EMBEDDING_MODEL_NAME
except ModuleNotFoundError:
    # This fallback is if the tests are run from the global repo root
    # and MyRAGProject is a subdirectory.
    sys.path.insert(0, os.path.join(os.getcwd(), "MyRAGProject"))
    from src.core import EmbeddingModel
    from src.config import EMBEDDING_MODEL_NAME


# Expected dimension for the default model
# You might want to fetch this programmatically from the model in a fixture
# if you plan to test with multiple models.
EXPECTED_DIMENSION = 384 # For 'sentence-transformers/all-MiniLM-L6-v2'

@pytest.fixture(scope="module")
def embedding_model():
    """Fixture to initialize the EmbeddingModel once per test module."""
    # Ensure .env can be found if tests are run from MyRAGProject directory
    # dotenv.load_dotenv(os.path.join(PROJECT_ROOT_FROM_TEST, ".env"))
    # No, config.py already calls load_dotenv(), assuming .env is in the CWD
    # when config.py is loaded.
    # If running tests from MyRAGProject, .env in MyRAGProject will be found.
    model = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    assert model.model is not None, "Failed to load the sentence transformer model."
    return model

def test_model_loading_and_dimension(embedding_model):
    """Tests if the model loads and reports the correct dimension."""
    assert embedding_model.model is not None
    dimension = embedding_model.get_embedding_dimension()
    assert dimension == EXPECTED_DIMENSION, \
        f"Model {EMBEDDING_MODEL_NAME} expected dimension {EXPECTED_DIMENSION}, got {dimension}"

def test_single_embedding_generation(embedding_model):
    """Tests generating an embedding for a single query."""
    sample_text = "This is a test sentence."
    embedding = embedding_model.embed_query(sample_text)

    assert embedding is not None, "Embedding should not be None."
    assert isinstance(embedding, list), "Embedding should be a list."
    assert len(embedding) == EXPECTED_DIMENSION, \
        f"Embedding dimension mismatch. Expected {EXPECTED_DIMENSION}, got {len(embedding)}."
    assert all(isinstance(x, float) for x in embedding), "All elements in embedding should be floats."

def test_batch_embedding_generation(embedding_model):
    """Tests generating embeddings for a batch of documents."""
    sample_texts = [
        "First sentence for batch processing.",
        "Second sentence, slightly different.",
        "And a third one to make it a batch."
    ]
    embeddings = embedding_model.embed_documents(sample_texts)

    assert embeddings is not None, "Embeddings should not be None."
    assert isinstance(embeddings, list), "Embeddings should be a list."
    assert len(embeddings) == len(sample_texts), \
        f"Number of embeddings ({len(embeddings)}) should match number of input texts ({len(sample_texts)})."

    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, list), f"Embedding {i} should be a list."
        assert len(embedding) == EXPECTED_DIMENSION, \
            f"Embedding {i} dimension mismatch. Expected {EXPECTED_DIMENSION}, got {len(embedding)}."
        assert all(isinstance(x, float) for x in embedding), \
            f"All elements in embedding {i} should be floats."

def test_empty_input_embed_documents(embedding_model):
    """Tests embed_documents with empty list input."""
    embeddings = embedding_model.embed_documents([])
    assert embeddings == [], "Embedding an empty list should return an empty list."

def test_empty_string_embed_query(embedding_model):
    """Tests embed_query with an empty string."""
    embedding = embedding_model.embed_query("")
    assert embedding is not None
    assert len(embedding) == EXPECTED_DIMENSION, \
        f"Embedding dimension mismatch for empty string. Expected {EXPECTED_DIMENSION}, got {len(embedding)}."

if __name__ == "__main__":
    # This allows running the test file directly for debugging, e.g., python tests/test_embedding.py
    # Note: Pytest is the recommended way to run tests.
    # You might need to adjust sys.path or run `pytest` from the `MyRAGProject` directory.
    
    # Example of how to run specific tests with pytest arguments:
    # pytest.main(["-v", __file__])
    
    # For direct run, simulate fixture manually if needed, or rely on pytest discovery.
    print("Running tests (direct execution, pytest is recommended)...")
    
    # A simple way to run all tests in this file if run directly:
    # This is not a replacement for pytest but can be useful for quick checks.
    _model = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    if _model.model:
        # Manually call test functions if needed for direct script execution
        # This is generally not how pytest tests are run.
        # Pytest handles test discovery and execution.
        print(f"Manually testing with model: {EMBEDDING_MODEL_NAME}, Dim: {_model.get_embedding_dimension()}")
        
        # Simulating fixture for direct run
        mock_model_fixture = _model

        test_model_loading_and_dimension(mock_model_fixture)
        test_single_embedding_generation(mock_model_fixture)
        test_batch_embedding_generation(mock_model_fixture)
        test_empty_input_embed_documents(mock_model_fixture)
        test_empty_string_embed_query(mock_model_fixture)
        print("Direct execution tests completed.")
    else:
        print("Failed to load model for direct execution tests.")
