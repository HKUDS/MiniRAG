# LocalRAG: A RAG Pipeline with Local Models

## Overview

LocalRAG is a Python-based Retrieval Augmented Generation (RAG) system designed to run entirely with locally hosted models. Inspired by projects like MiniRAG, this system aims to provide a foundational RAG pipeline using local sentence transformer models for embeddings and local Large Language Models (LLMs) from the Hugging Face `transformers` library for text generation. This approach allows for greater privacy, control, and offline usability.

The project demonstrates loading text data, chunking it, generating embeddings, storing/retrieving document chunks (currently placeholder retrieval), and generating answers to queries using a local LLM based on provided context.

## Features

-   **Local Embedding Generation**: Utilizes `sentence-transformers` library to generate dense vector embeddings for text data locally.
-   **Local LLM for Generation**: Employs Hugging Face `transformers` library to load and use local LLMs for generating responses.
-   **Basic RAG Pipeline**: Implements a simple pipeline involving data processing, (placeholder) retrieval, prompt construction, and LLM-based generation.
-   **Configurable Models**: Allows easy configuration of embedding and LLM models through `src/config.py`.
-   **Modular Design**: Core components like data processing, embedding, vector database interaction (placeholder), and LLM interface are separated for clarity.

## Directory Structure

-   `MyRAGProject/`: Root directory of the project.
    -   `data/`: Intended for storing input data files (e.g., `.txt` files). Contains `sample.txt` for demonstration.
    -   `models/`: Intended for storing model-related files, such as FAISS indexes or other local model artifacts (currently used for placeholder vector DB path).
    -   `src/`: Contains the main source code for the RAG application.
        -   `__init__.py`: Makes `src` a Python package.
        -   `config.py`: Handles configuration settings (e.g., model names, paths).
        -   `core.py`: Defines core components like `DataProcessor`, `EmbeddingModel`, `VectorDatabase`, `LLMInterface`, and `RAGSystem`.
        -   `main.py`: Main script to run the RAG application.
        -   `utils.py`: For utility functions (currently basic).
    -   `tests/`: Contains all Pytest test files for the project.
        -   `__init__.py`: Makes `tests` a Python package.
        -   `test_data_processing.py`: Tests for data loading and chunking.
        -   `test_embedding.py`: Tests for the local embedding model.
        -   `test_llm.py`: Tests for the local LLM interface.
        -   `test_rag_pipeline.py`: Integration tests for the RAG pipeline.
    -   `requirements.txt`: Lists project dependencies.
    -   `.env.example`: Example environment file template.
    -   `README.md`: This file.

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd MyRAGProject
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model Downloads**:
    The Hugging Face `transformers` and `sentence-transformers` libraries will automatically download the specified pre-trained models (e.g., for embeddings and LLM) on their first use. These models are typically stored in the Hugging Face cache directory (e.g., `~/.cache/huggingface/hub/` or `~/.cache/huggingface/sentence_transformers/`). Ensure you have an internet connection for the initial download.

5.  **Environment Variables** (Optional):
    If you plan to use specific configurations not suitable for direct inclusion in `config.py` (e.g., API keys for future extensions, or overriding default paths via environment variables), you can:
    -   Copy `.env.example` to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    -   Edit the `.env` file to set your desired variables. `src/config.py` is set up to load variables from this file. For the current fully local setup, this might not be strictly necessary unless you override default model names or paths.

## How to Run

1.  **Place Data**:
    -   Input text files (e.g., `.txt`) should be placed in the `MyRAGProject/data/` directory.
    -   A `sample.txt` file is already provided for demonstration.

2.  **Run the Main Script**:
    Execute the main application script from the `MyRAGProject` root directory:
    ```bash
    python src/main.py
    ```

3.  **Expected Output/Behavior**:
    -   The script will initialize the RAG components (DataProcessor, EmbeddingModel, VectorDatabase, LLMInterface).
    -   It will load and process the data from `MyRAGProject/data/sample.txt`.
    -   It will "build" an index using the processed documents (currently, this involves generating embeddings if possible and storing documents for placeholder search).
    -   It will then process a sample query defined in `src/main.py` (e.g., "What is crucial for retrieval accuracy?").
    -   The RAG system will attempt to retrieve relevant context (using placeholder keyword search) and generate a response using the local LLM.
    -   You will see print statements indicating these steps, including model loading attempts, data processing, and the final query and response.
    -   **Note**: If the local models (embedding or LLM) fail to load due to environment issues (like insufficient disk space for PyTorch), the script will print error messages and skip the query processing step.

## Configuration

-   Core configurations are managed in `MyRAGProject/src/config.py`.
-   You can change the default local models by modifying the following variables in `src/config.py` or by setting them as environment variables (which `config.py` will load via `python-dotenv` if a `.env` file is present):
    -   `EMBEDDING_MODEL_NAME`: Specifies the sentence transformer model for embeddings (default: `"sentence-transformers/all-MiniLM-L6-v2"`).
    -   `LLM_MODEL_NAME`: Specifies the Hugging Face model for the LLM (default: `"distilgpt2"`).
-   Other paths, like `VECTOR_DB_PATH`, `RAW_DATA_DIR`, etc., can also be configured there.

## Testing

-   To run the test suite (requires `pytest`):
    ```bash
    pytest MyRAGProject/tests/
    ```
    Or, from within the `MyRAGProject` directory:
    ```bash
    python -m pytest tests/
    ```

-   **Important Note on Test Execution**:
    The project's tests rely on libraries like `torch`, `sentence-transformers`, and `transformers`. These libraries, especially `torch`, can be very large. In constrained environments (like some sandboxed CI/CD runners or low-resource machines), installation of these dependencies might fail due to insufficient disk space. This can lead to `ImportError` (e.g., `ImportError: cannot import name 'Tensor' from 'torch'`) during test collection or execution, causing tests to fail or not run at all. If you encounter such issues, it's likely an environmental limitation rather than a bug in the project code itself.

## Future Improvements

-   **Support for More Data Types**: Extend `DataProcessor` to handle PDFs, DOCX, URLs, etc.
-   **Advanced Vector Search**: Replace the placeholder keyword search with a proper vector database implementation (e.g., using FAISS for efficient similarity search).
-   **Improved Chunking Strategies**: Implement more sophisticated text chunking methods (e.g., recursive character splitting, token-based chunking).
-   **UI/API Interface**: Develop a simple web interface (e.g., using Flask/Streamlit) or an API for easier interaction with the RAG system.
-   **Batch Processing**: Add capabilities for processing multiple queries or documents in batch.
-   **Evaluation Framework**: Integrate an evaluation framework to measure retrieval and generation quality.
-   **More Robust Model Error Handling**: Enhance error handling and fallbacks for model loading and generation.

This README should provide a good overview and guide for users of the LocalRAG project.Okay, I have created the `README.md` file for the `MyRAGProject`.

The content includes:
-   **Project Title**: "LocalRAG: A RAG Pipeline with Local Models"
-   **Overview**: Description of the project's purpose.
-   **Features**: Key capabilities like local embeddings, local LLM, basic RAG pipeline, and configurability.
-   **Directory Structure**: Explanation of `data`, `models`, `src`, `tests`, and key files.
-   **Setup Instructions**: Steps for cloning, virtual environment, dependencies, model downloads, and `.env` usage.
-   **How to Run**: Instructions on placing data, running `src/main.py`, and expected behavior.
-   **Configuration**: Explanation of `src/config.py` for model selection.
-   **Testing**: How to run tests with `pytest`, including the crucial note about potential failures due to environmental disk space limitations affecting PyTorch installation.
-   **Future Improvements**: Suggestions for enhancing the project.

The file is placed in `MyRAGProject/README.md` as requested.
