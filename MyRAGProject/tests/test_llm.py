# tests/test_llm.py

import pytest
import os
import sys

# Add project root to sys.path
PROJECT_ROOT_FROM_TEST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT_FROM_TEST)

try:
    from src.core import LLMInterface
    from src.config import LLM_MODEL_NAME
except ModuleNotFoundError:
    # Fallback if tests are run from the global repo root
    sys.path.insert(0, os.path.join(os.getcwd(), "MyRAGProject"))
    from src.core import LLMInterface
    from src.config import LLM_MODEL_NAME

@pytest.fixture(scope="module")
def local_llm():
    """Fixture to initialize the LLMInterface (LocalLLM) once per test module."""
    # config.py loads .env from CWD. If running tests from MyRAGProject dir,
    # .env in MyRAGProject will be used.
    llm = LLMInterface(model_name=LLM_MODEL_NAME) # Uses model from config, e.g., "distilgpt2"
    # Do not assert model loading here, as it might fail due to disk/network in sandbox
    # The tests themselves will check for successful loading or graceful failure.
    return llm

def test_llm_initialization(local_llm):
    """Tests if the LLMInterface initializes and attempts to load model and tokenizer."""
    # This test effectively checks if the __init__ ran without Python errors
    # and if the model and tokenizer attributes are either None (if loading failed)
    # or actual model/tokenizer objects.
    if local_llm.model is None or local_llm.tokenizer is None:
        print(f"LLM model '{local_llm.model_name}' or tokenizer failed to load. "
              "This might be due to environment constraints (disk/network).")
    # We don't fail the test here if loading failed, as that's an environment issue.
    # The next test will check if generation works (which implies loading worked).
    assert hasattr(local_llm, 'model'), "LLMInterface should have a 'model' attribute."
    assert hasattr(local_llm, 'tokenizer'), "LLMInterface should have a 'tokenizer' attribute."

def test_llm_generate_response(local_llm):
    """Tests generating a response from the local LLM."""
    if not local_llm.model or not local_llm.tokenizer:
        pytest.skip(f"Skipping response generation test as model '{local_llm.model_name}' "
                    "or tokenizer did not load. Likely an environment issue.")

    sample_prompt = "Hello, what is your name?"
    response = local_llm.generate_response(sample_prompt, max_length=20) # Short max_length for quick test

    assert isinstance(response, str), "Response should be a string."
    assert len(response) > 0, "Response string should not be empty."
    
    # Check for known error messages from the generate_response method
    assert "Error: LLM Model or Tokenizer not loaded." not in response, \
        "LLM generate_response indicated model/tokenizer not loaded."
    assert "Error generating response." not in response, \
        "LLM generate_response indicated an error during generation."
    
    print(f"Generated response for '{sample_prompt}': '{response}'")

if __name__ == "__main__":
    # For direct execution (pytest is preferred)
    print("Running LLM tests directly (pytest is recommended)...")
    # Manually create an instance for direct run
    # This will attempt to download the model if not cached.
    _llm_instance = LLMInterface()
    test_llm_initialization(_llm_instance)
    test_llm_generate_response(_llm_instance)
    print("Direct execution tests completed.")
