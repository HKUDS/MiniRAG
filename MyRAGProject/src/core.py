# core.py
# Contains core RAG components like data loading, vectorization, and querying

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME # Use .config for relative import

class EmbeddingModel:
    """Handles loading and using a sentence transformer model for embeddings."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Sentence transformer model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading sentence transformer model '{model_name}': {e}")
            self.model = None

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of documents."""
        if self.model:
            print(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.model.encode(documents, show_progress_bar=False) # Or True for progress
            print("Embeddings generated.")
            return embeddings.tolist() # Convert numpy arrays to lists of floats
        return []

    def embed_query(self, query: str) -> list[float]:
        """Generates embedding for a single query."""
        if self.model:
            print(f"Generating embedding for query: '{query}'")
            embedding = self.model.encode(query, show_progress_bar=False)
            print("Query embedding generated.")
            return embedding.tolist() # Convert numpy array to list of floats
        return []

    def get_embedding_dimension(self) -> int:
        """Returns the dimension of the embeddings."""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return -1 # Or raise an error

import os # For filepath operations in DataProcessor

class DataProcessor:
    def __init__(self):
        # data_path can be used as a default base directory if needed,
        # but load_and_process_data will take specific filepaths.
        print("DataProcessor initialized.")

    def load_and_process_data(self, filepath: str) -> list[str]:
        """
        Loads data from the given filepath, processes it, and chunks it.
        Currently supports .txt files and chunks by paragraph.
        """
        print(f"Attempting to load and process data from: {filepath}")
        chunks = []
        try:
            file_extension = os.path.splitext(filepath)[1].lower()
            if file_extension == ".txt":
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Simple chunking: split by paragraph
                raw_chunks = text.split('\n\n')
                chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
                print(f"Successfully processed {filepath}. Found {len(chunks)} chunks.")
            # TODO: Add support for other file types like PDF
            # elif file_extension == ".pdf":
            #     try:
            #         import PyPDF2
            #         with open(filepath, 'rb') as f:
            #             reader = PyPDF2.PdfReader(f)
            #             text_content = ""
            #             for page_num in range(len(reader.pages)):
            #                 text_content += reader.pages[page_num].extract_text() or ""
            #         # Further chunking would be needed for PDF text_content
            #         raw_chunks = text_content.split('\n\n') # Example, might need refinement
            #         chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
            #         print(f"Successfully processed PDF {filepath}. Found {len(chunks)} chunks.")
            #     except ImportError:
            #         print("PyPDF2 library is not installed. Please install it to process PDF files.")
            #     except Exception as e:
            #         print(f"Error processing PDF file {filepath}: {e}")
            else:
                print(f"Unsupported file type: {file_extension} for {filepath}. Only .txt is currently supported.")
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        except Exception as e:
            print(f"An error occurred while processing {filepath}: {e}")
        
        return chunks

class VectorDatabase:
    def __init__(self, index_path=None):
        self.index_path = index_path
        self.embedding_model = EmbeddingModel() # Instantiate our embedding model
        # TODO: Initialize FAISS or other vector DB (e.g., self.index = faiss.IndexFlatL2(...))

    def build_index(self, documents: list[str]):
        """Builds a vector index from the given documents."""
        if not self.embedding_model or not self.embedding_model.model:
            print("Error: Embedding model not loaded. Cannot build index.")
            return

        doc_embeddings = self.embedding_model.embed_documents(documents)
        if not doc_embeddings:
            print("Error: No embeddings generated by EmbeddingModel. Cannot build index.")
            return
        
        # TODO: Implement actual FAISS or other vector DB index building
        # Example with FAISS:
        # import faiss
        # import numpy as np
        # dimension = self.embedding_model.get_embedding_dimension()
        # if dimension > 0 and self.embedding_model.model: # Check model loaded
        #     self.index = faiss.IndexFlatL2(dimension)
        #     embeddings_np = np.array(doc_embeddings).astype('float32')
        #     self.index.add(embeddings_np)
        #     print(f"FAISS index built with {self.index.ntotal} vectors of dimension {dimension}.")
        # else:
        #     print("Error: Could not get embedding dimension or model not loaded. Cannot build FAISS index.")
        # For now, just acknowledge the intention
        self.documents_for_search = documents # Store original documents for placeholder search
        print(f"Vector index building process initiated for {len(doc_embeddings)} document embeddings. Actual indexing is a TODO.")
        print(f"Stored {len(documents)} original document chunks for placeholder search.")


    def search(self, query: str, k: int = 5) -> list[str]:
        """
        Searches the vector index for documents similar to the query.
        Currently returns placeholder content or simple keyword match on stored documents.
        """
        if not self.embedding_model or not self.embedding_model.model:
            print("Error: Embedding model not loaded. Cannot perform search.")
            return []
        
        query_vector = self.embedding_model.embed_query(query)
        if not query_vector:
            print("No query embedding generated. Cannot perform search.")
            return []

        # TODO: Implement actual similarity search with FAISS or other vector DB
        # Example with FAISS:
        # import numpy as np
        # if self.index and query_vector:
        #     query_vector_np = np.array([query_vector]).astype('float32')
        #     distances, indices = self.index.search(query_vector_np, k)
        #     # Return the actual documents based on indices
        #     # results = [self.documents_for_search[i] for i in indices[0]]
        #     # print(f"FAISS search found indices: {indices[0]} for query: '{query}'")
        #     # return results 
        # else:
        #     print("FAISS Index not built or query_vector missing. Cannot perform FAISS search.")

        # Placeholder search: simple keyword matching on stored documents
        if hasattr(self, 'documents_for_search') and self.documents_for_search:
            print(f"Performing placeholder keyword search for '{query}' in {len(self.documents_for_search)} documents.")
            results = [doc for doc in self.documents_for_search if query.lower() in doc.lower()]
            print(f"Placeholder search found {len(results)} documents.")
            return results[:k]
        
        print(f"Searching for top {k} similar documents (actual search is a TODO). Query vector generated if model loaded.")
        return ["Placeholder search result 1: content related to " + query, 
                "Placeholder search result 2: more details about " + query] # Placeholder for search results

class LLMInterface:
    def __init__(self, model_name):
        self.model_name = model_name
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import LLM_MODEL_NAME

class LLMInterface: # This will now be our LocalLLM
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        try:
            print(f"Loading Hugging Face model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            # Model loading can fail due to various reasons including network or disk space for model cache

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generates a response from the LLM given a prompt."""
        if not self.model or not self.tokenizer:
            return "Error: LLM Model or Tokenizer not loaded."
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512) # Ensure max_length for tokenizer
            
            # Ensure model is on CPU if no GPU is explicitly handled. Forcing CPU to avoid potential issues in sandbox.
            # device = "cuda" if torch.cuda.is_available() else "cpu" 
            # self.model.to(device)
            # inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate output tokens
            # Pad token ID is crucial for open-ended generation with padding.
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length + len(inputs["input_ids"][0]), # max_length relative to prompt length
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=2, # Optional: to prevent repetitive text
                early_stopping=True     # Optional: to stop generation earlier
            )
            
            response = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Clean up the prompt from the beginning of the response if model includes it
            if response.startswith(prompt):
                response = response[len(prompt):].lstrip()
                
            return response
        except Exception as e:
            print(f"Error during LLM response generation: {e}")
            return "Error generating response."

class RAGSystem:
    def __init__(self, data_processor, vector_db, llm_interface):
        self.data_processor = data_processor
        self.vector_db = vector_db
        self.llm_interface = llm_interface

    def process_query(self, query: str):
        """Processes a query using the RAG pipeline."""
        print(f"Processing query: {query}")
        
        # 1. Retrieve relevant documents (simplified)
        # In a real system, documents would be pre-indexed.
        # Here, we might simulate retrieving some context or just use the query directly.
        # For now, let's assume search_results are the context documents.
        # In a full RAG, documents would be retrieved based on the query.
        # Let's simulate some context retrieval for now:
        # documents = self.data_processor.load_and_process_data() # This might be too slow for each query
        # self.vector_db.build_index(documents) # Indexing should be done beforehand
        retrieved_docs_content = self.vector_db.search(query) # This returns placeholder indices for now

        # For the purpose of this task, VectorDatabase.search returns a list of strings (placeholder)
        # If it returned actual document content, we'd use that.
        # Let's assume `retrieved_docs_content` is a list of strings if search is implemented,
        # or an empty list if not.
        
        # 2. Construct a prompt for the LLM
        # This is a simple way to combine query and context.
        # More sophisticated prompt engineering would be needed for better results.
        if retrieved_docs_content: # Assuming search_results are strings of content
            context_str = "\n\n".join(retrieved_docs_content)
            prompt = f"Based on the following context:\n{context_str}\n\nAnswer the query: {query}"
        else:
            # Fallback if no context is retrieved or search is not yet functional
            prompt = f"Answer the query: {query}"

        print(f"Generated prompt for LLM: {prompt[:200]}...") # Print start of prompt

        # 3. Generate response using the LLM
        response = self.llm_interface.generate_response(prompt)
        
        print(f"LLM generated response: {response[:200]}...") # Print start of response
        return response
