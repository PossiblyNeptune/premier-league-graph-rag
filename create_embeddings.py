"""
Document Processor with Football Tactics Domain Preprocessing
Creates embeddings and vector store using HuggingFace embeddings.
"""

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import os
from dotenv import load_dotenv

from football_tactics_preprocessor import FootballTacticsPreprocessor

load_dotenv()


class DocumentProcessor:
    def __init__(self, chunk_size=1500, chunk_overlap=250):
        """
        Initializes the DocumentProcessor with HuggingFace embeddings.

        Args:
        - chunk_size (int): The size of each text chunk. Default is 1500.
        - chunk_overlap (int): The overlap between consecutive chunks. Default is 250.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print("Loading HuggingFace embeddings model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ HuggingFace embeddings loaded successfully")
        
        self.preprocessor = FootballTacticsPreprocessor()
        print("📋 DocumentProcessor initialized with HuggingFace embeddings")

    def process_documents(self, documents):
        """
        Processes documents by splitting, preprocessing, and creating embeddings.

        Args:
        - documents (list): A list of documents to be processed.

        Returns:
        - tuple: (splits, vector_store)
        """
        print("\nSplitting documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(splits)} document chunks.")
        
        print("Applying football preprocessing...")
        for i, split in enumerate(splits):
            processed_content, tactical_entities = self.preprocessor.preprocess_chunk(
                split.page_content
            )
            
            split.page_content = processed_content
            split.metadata['chunk_id'] = i
            split.metadata['source_file'] = split.metadata.get('source', 'unknown')
            split.metadata['page_number'] = split.metadata.get('page', 'unknown')
            split.metadata['chunk_length'] = len(split.page_content)
            
            # Store all extracted entities
            split.metadata['formations'] = tactical_entities.get('formations', [])
            split.metadata['tactical_roles'] = tactical_entities.get('tactical_roles', [])
            split.metadata['defensive_concepts'] = tactical_entities.get('defensive_concepts', [])
            split.metadata['offensive_concepts'] = tactical_entities.get('offensive_concepts', [])
            split.metadata['key_figures'] = tactical_entities.get('key_figures', [])  # NEW
            split.metadata['key_teams'] = tactical_entities.get('key_teams', [])  # NEW
            
            tactical_keywords = FootballTacticsPreprocessor.create_tactical_keywords(tactical_entities)
            split.metadata['tactical_keywords'] = tactical_keywords
        
        print(f"✓ Applied preprocessing to {len(splits)} chunks")
        
        print("Creating vector store with embeddings...")
        vector_store = FAISS.from_documents(splits, self.embeddings)
        print("✓ Vector store created successfully!")
        
        return splits, vector_store


    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Computes a cosine similarity matrix for embeddings."""
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        print(f"✓ Similarity matrix computed with shape {similarity_matrix.shape}.")
        return similarity_matrix

    def save_vector_store(self, vector_store, path: str = "vector_store"):
        """Saves the FAISS vector store to disk."""
        print(f"Saving vector store to {path}...")
        vector_store.save_local(path)
        print("✓ Vector store saved successfully.")

    def load_vector_store(self, path: str = "vector_store"):
        """Loads a FAISS vector store from disk."""
        print(f"Loading vector store from {path}...")
        vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Vector store loaded successfully.")
        return vector_store

    def add_documents_to_vector_store(self, vector_store, documents):
        """Adds new documents to vector store with preprocessing."""
        print(f"Adding {len(documents)} new documents to vector store...")
        splits = self.text_splitter.split_documents(documents)
        
        for split in splits:
            processed_content, tactical_entities = self.preprocessor.preprocess_chunk(
                split.page_content
            )
            split.page_content = processed_content
            
            # Store all extracted entities
            split.metadata['formations'] = tactical_entities.get('formations', [])
            split.metadata['tactical_roles'] = tactical_entities.get('tactical_roles', [])
            split.metadata['defensive_concepts'] = tactical_entities.get('defensive_concepts', [])
            split.metadata['offensive_concepts'] = tactical_entities.get('offensive_concepts', [])
            split.metadata['key_figures'] = tactical_entities.get('key_figures', [])  # NEW
            split.metadata['key_teams'] = tactical_entities.get('key_teams', [])  # NEW
            
            tactical_keywords = FootballTacticsPreprocessor.create_tactical_keywords(
                tactical_entities
            )
            split.metadata['tactical_keywords'] = tactical_keywords
        
        vector_store.add_documents(splits)
        print("✓ Documents added successfully.")
        return vector_store


    def get_embedding_dimension(self) -> int:
        """Returns the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return 384


# Example usage and testing
if __name__ == "__main__":
    from langchain_community.document_loaders import PyPDFLoader
    
    print("="*60)
    print("Testing DocumentProcessor with HuggingFace Embeddings")
    print("="*60)
    
    processor = DocumentProcessor()
    test_path = "data/The_Mixer.pdf"
    
    if os.path.exists(test_path):
        print(f"\nLoading document from {test_path}...")
        loader = PyPDFLoader(test_path)
        documents = loader.load()
        documents = documents[:5]  # Test with first 5 pages
        
        print(f"Loaded {len(documents)} pages.")
        splits, vector_store = processor.process_documents(documents)
        
        print("\n" + "="*60)
        print("Sample Processed Chunks:")
        print("="*60)
        for i, split in enumerate(splits[:2]):
            print(f"\nChunk {i}:")
            print(f"  Length: {split.metadata['chunk_length']}")
            print(f"  Formations: {split.metadata['formations']}")
            print(f"  Content preview: {split.page_content[:200]}...")
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
    else:
        print(f"\nTest file not found: {test_path}")
