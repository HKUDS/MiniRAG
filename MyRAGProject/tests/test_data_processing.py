# tests/test_data_processing.py

import pytest
import os
import sys

# Add project root to sys.path
PROJECT_ROOT_FROM_TEST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT_FROM_TEST) # Add MyRAGProject to path

try:
    from src.core import DataProcessor, VectorDatabase, EmbeddingModel
    from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH
    # from src.main import load_and_process_data_into_db # This function is in main for orchestration
                                                       # For testing units, better to test DataProcessor directly
except ModuleNotFoundError:
    # Fallback if tests are run from the global repo root
    sys.path.insert(0, os.path.join(os.getcwd(), "MyRAGProject"))
    from src.core import DataProcessor, VectorDatabase, EmbeddingModel
    from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH
    # from src.main import load_and_process_data_into_db


# Sample text content matching MyRAGProject/data/sample.txt
SAMPLE_TEXT_CONTENT = """This is the first paragraph of our sample text file. It contains a few sentences to demonstrate the loading and processing capabilities of the RAG system. We aim to chunk this text into meaningful segments.

The second paragraph provides more content. RAG systems often benefit from well-defined document chunks. These chunks are then vectorized and stored in a database for efficient retrieval. Proper chunking strategy is crucial for retrieval accuracy.

Finally, the third paragraph concludes this sample document. It's a short document, but sufficient for initial testing of the data processing pipeline. Future enhancements could include handling various file formats like PDF, DOCX, or even web URLs.
The RAG model will use these chunks to find relevant information."""
EXPECTED_CHUNKS = [
    "This is the first paragraph of our sample text file. It contains a few sentences to demonstrate the loading and processing capabilities of the RAG system. We aim to chunk this text into meaningful segments.",
    "The second paragraph provides more content. RAG systems often benefit from well-defined document chunks. These chunks are then vectorized and stored in a database for efficient retrieval. Proper chunking strategy is crucial for retrieval accuracy.",
    "Finally, the third paragraph concludes this sample document. It's a short document, but sufficient for initial testing of the data processing pipeline. Future enhancements could include handling various file formats like PDF, DOCX, or even web URLs.\nThe RAG model will use these chunks to find relevant information."
]


@pytest.fixture(scope="module")
def data_processor():
    return DataProcessor()

@pytest.fixture(scope="module")
def sample_txt_filepath(tmp_path_factory):
    # Create a temporary sample file for tests to avoid relying on git-tracked file state during test
    # tmp_path_factory is a session-scoped fixture, so we create a subdirectory for this module
    data_dir = tmp_path_factory.mktemp("data_processing_data")
    filepath = data_dir / "sample_test.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TEXT_CONTENT)
    return str(filepath) # Return as string, as DataProcessor expects str path

def test_load_and_process_txt_file(data_processor, sample_txt_filepath):
    """Tests loading and paragraph-chunking of a .txt file."""
    chunks = data_processor.load_and_process_data(sample_txt_filepath)
    
    assert chunks is not None, "Processed data should not be None."
    assert isinstance(chunks, list), "Processed data should be a list."
    assert len(chunks) == len(EXPECTED_CHUNKS), \
        f"Expected {len(EXPECTED_CHUNKS)} chunks, got {len(chunks)}."
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, str), f"Chunk {i} should be a string."
        assert chunk == EXPECTED_CHUNKS[i], f"Chunk {i} content mismatch."

def test_unsupported_file_type(data_processor, tmp_path):
    """Tests behavior with an unsupported file type."""
    unsupported_filepath = tmp_path / "sample.docx"
    with open(unsupported_filepath, "w") as f:
        f.write("This is a docx file.")
    chunks = data_processor.load_and_process_data(str(unsupported_filepath))
    assert chunks == [], "Should return empty list for unsupported file type."

def test_file_not_found(data_processor):
    """Tests behavior when a file is not found."""
    chunks = data_processor.load_and_process_data("non_existent_file.txt")
    assert chunks == [], "Should return empty list if file not found."

def test_data_processing_and_indexing_flow(data_processor, sample_txt_filepath):
    """
    Tests the flow of loading data, processing it, and passing it to VectorDatabase.
    Focuses on the data flow rather than actual embedding/indexing success.
    """
    # 1. Initialize components (VectorDatabase internally initializes EmbeddingModel)
    # We pass a specific embedding model name for consistency if needed,
    # but default from config should be fine.
    vector_db = VectorDatabase(index_path=os.path.join(PROJECT_ROOT_FROM_TEST, "models", "test_db.faiss"))

    # 2. Load and process data using DataProcessor
    processed_documents = data_processor.load_and_process_data(sample_txt_filepath)
    
    assert processed_documents is not None
    assert len(processed_documents) == len(EXPECTED_CHUNKS)

    # 3. "Build index" in VectorDatabase
    # This step will attempt to generate embeddings if the model loaded.
    # We are primarily testing that the data flows correctly into build_index.
    vector_db.build_index(processed_documents)

    # Check if documents were passed to VectorDatabase (for placeholder search)
    assert hasattr(vector_db, 'documents_for_search'), \
        "VectorDatabase should have 'documents_for_search' after build_index."
    assert vector_db.documents_for_search is not None
    assert len(vector_db.documents_for_search) == len(EXPECTED_CHUNKS), \
        "Stored documents in VectorDB do not match processed documents."
    assert vector_db.documents_for_search == EXPECTED_CHUNKS

    # Further checks could involve mocking EmbeddingModel if we want to isolate VectorDB logic
    # without actual embedding generation, especially if environment issues persist.
    # For now, this confirms the data pipeline up to the point of potential embedding.
    if not vector_db.embedding_model.model:
        print("Test info: Embedding model did not load during this test run (likely environment issue). "
              "Data flow up to embedding generation is being checked.")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
