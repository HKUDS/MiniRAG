# main.py
# Main script to run the RAG application

from src.core import DataProcessor, VectorDatabase, LLMInterface, RAGSystem, EmbeddingModel
from src.config import VECTOR_DB_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, RAW_DATA_DIR
import os

def initialize_rag_system():
    """Initializes all components of the RAG system."""
    print("Initializing RAG components...")
    # EmbeddingModel is used by VectorDatabase internally
    # No need to pass it explicitly if VectorDatabase instantiates it.
    # embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    
    data_processor = DataProcessor()
    vector_db = VectorDatabase(index_path=VECTOR_DB_PATH) # EmbeddingModel is created inside VectorDatabase
    llm_interface = LLMInterface(model_name=LLM_MODEL_NAME)
    
    rag_system = RAGSystem(
        data_processor=data_processor,
        vector_db=vector_db,
        llm_interface=llm_interface
    )
    print("RAG components initialized.")
    return rag_system

def load_and_process_data_into_db(
    data_processor: DataProcessor, 
    vector_db: VectorDatabase, 
    filepath: str
):
    """
    Loads data from the given filepath using DataProcessor,
    processes it, and then builds the index in VectorDatabase.
    """
    print(f"\n--- Loading and Processing Data for: {filepath} ---")
    processed_documents = data_processor.load_and_process_data(filepath)
    
    if processed_documents:
        print(f"Data loaded and processed. Found {len(processed_documents)} chunks.")
        print("Building vector database index with these documents...")
        vector_db.build_index(processed_documents) # Builds index using internal EmbeddingModel
        print("Vector database index building process completed (or initiated if async).")
    else:
        print(f"No documents were processed from {filepath}. Index not built.")

def main():
    print("--- Starting RAG Application ---")
    
    # Initialize the RAG system
    rag_system = initialize_rag_system()

    # Define the path to the sample data file
    # Assuming RAW_DATA_DIR is 'data/raw/' and sample.txt is directly in 'data/'
    # For this example, let's place sample.txt in 'MyRAGProject/data/'
    # and adjust config.py or path logic accordingly if needed.
    # For now, construct path relative to project root (MyRAGProject)
    # PROJECT_ROOT for main.py would be parent of src, i.e., MyRAGProject
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    # This assumes main.py is in MyRAGProject/src. So parent is MyRAGProject.
    # If config.PROJECT_ROOT is reliable, use that.
    # from src.config import PROJECT_ROOT # This could also be used.
    
    sample_file_path = os.path.join(project_root, "data", "sample.txt") 
    # Corrected path assuming data/sample.txt relative to MyRAGProject root

    # Load, process data, and build the vector database index
    # We need the DataProcessor and VectorDatabase instances from the RAGSystem
    load_and_process_data_into_db(
        data_processor=rag_system.data_processor,
        vector_db=rag_system.vector_db,
        filepath=sample_file_path
    )

    # Example: Process a sample query
    if rag_system.llm_interface.model and rag_system.vector_db.embedding_model.model:
        print("\n--- Processing a Sample Query ---")
        sample_query = "What is crucial for retrieval accuracy?"
        # sample_query = "paragraph" # To test keyword search
        response = rag_system.process_query(sample_query)
        print(f"\nQuery: {sample_query}")
        print(f"Response: {response}")
    else:
        print("\nSkipping sample query processing as LLM or Embedding Model failed to load.")
        if not rag_system.llm_interface.model:
            print("Reason: LLMInterface model not loaded.")
        if not rag_system.vector_db.embedding_model.model:
            print("Reason: EmbeddingModel (in VectorDB) not loaded.")

    print("\n--- RAG Application Finished ---")

if __name__ == "__main__":
    main()
