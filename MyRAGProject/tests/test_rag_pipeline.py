# tests/test_rag_pipeline.py

import pytest
import os
import sys

# Add project root to sys.path
PROJECT_ROOT_FROM_TEST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT_FROM_TEST) # Add MyRAGProject to path

try:
    from src.main import initialize_rag_system, load_and_process_data_into_db
    from src.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT # To build sample file path
except ModuleNotFoundError:
    # Fallback if tests are run from the global repo root
    sys.path.insert(0, os.path.join(os.getcwd(), "MyRAGProject"))
    from src.main import initialize_rag_system, load_and_process_data_into_db
    from src.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT


@pytest.fixture(scope="module")
def initialized_rag_system():
    """
    Fixture to initialize the RAGSystem and load data once per test module.
    This is an integration test fixture.
    """
    print("\n--- (Fixture) Initializing RAG System for Integration Test ---")
    rag_system = initialize_rag_system()
    
    # Determine the path to sample.txt. CONFIG_PROJECT_ROOT should be MyRAGProject/
    # This assumes config.py's PROJECT_ROOT is correctly set to MyRAGProject base.
    sample_file_path = os.path.join(CONFIG_PROJECT_ROOT, "data", "sample.txt")
    
    if not os.path.exists(sample_file_path):
        # As a fallback if config.PROJECT_ROOT is tricky in test environment, try relative to this test file
        # This assumes test file is in MyRAGProject/tests/
        alt_sample_file_path = os.path.join(PROJECT_ROOT_FROM_TEST, "data", "sample.txt")
        if os.path.exists(alt_sample_file_path):
            sample_file_path = alt_sample_file_path
        else:
            pytest.fail(f"Sample data file not found at {sample_file_path} or {alt_sample_file_path}. "
                        "Ensure MyRAGProject/data/sample.txt exists.")

    print(f"--- (Fixture) Loading data from: {sample_file_path} ---")
    load_and_process_data_into_db(
        data_processor=rag_system.data_processor,
        vector_db=rag_system.vector_db,
        filepath=sample_file_path
    )
    print("--- (Fixture) RAG System Initialized and Data Loaded ---")
    return rag_system

def test_rag_system_integration(initialized_rag_system):
    """
    Tests the full RAG pipeline flow: query -> retrieve (placeholder) -> prompt -> generate.
    """
    rag_system = initialized_rag_system

    # Check if models loaded, otherwise skip. This is crucial due to environment issues.
    if not rag_system.llm_interface.model:
        pytest.skip("Skipping RAG integration test: LLM model not loaded.")
    if not rag_system.vector_db.embedding_model.model:
        pytest.skip("Skipping RAG integration test: Embedding model not loaded.")

    print("\n--- (Test) Processing Query via RAGSystem ---")
    # Query relevant to MyRAGProject/data/sample.txt
    # The sample.txt contains: "Proper chunking strategy is crucial for retrieval accuracy."
    sample_query = "What is crucial for retrieval accuracy?" 
    
    response = rag_system.process_query(sample_query)

    assert isinstance(response, str), "The response from RAGSystem should be a string."
    assert len(response) > 0, "The response string should not be empty."
    
    # Check for known error messages from the LLMInterface
    assert "Error: LLM Model or Tokenizer not loaded." not in response, \
        "RAG system's LLM indicated model/tokenizer not loaded."
    assert "Error generating response." not in response, \
        "RAG system's LLM indicated an error during generation."
        
    print(f"\nQuery: {sample_query}")
    print(f"Retrieved Context (from placeholder search in VectorDB):")
    # This requires VectorDB's search to actually return the docs it found for the prompt
    # The current placeholder search does this.
    # We can't easily access the exact context_str formed inside process_query without modifying it.
    # However, we can see the effect in the final response.
    
    print(f"Final Response: {response}")
    
    # A more advanced test would be to check if the response contains expected keywords
    # related to "chunking strategy" or "retrieval accuracy" based on the sample_query and sample.txt.
    # However, this depends heavily on the LLM's performance.
    # For now, a non-empty, non-error string is the primary assertion.
    # Example (very basic, might fail with weak LLMs):
    # assert "chunking" in response.lower() or "strategy" in response.lower(), \
    #     "Response doesn't seem to contain relevant keywords."

if __name__ == "__main__":
    # This allows running the test file directly.
    # Note: Pytest is the recommended way to run tests.
    pytest.main(["-v", __file__])
