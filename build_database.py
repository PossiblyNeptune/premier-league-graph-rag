"""
Database Builder with Enhanced Logging and Error Handling
Auto-scans PDFs and builds complete graph RAG database.
Uses LOCAL Ollama (no API needed).
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

from create_embeddings import DocumentProcessor
from create_database import KnowledgeGraph

load_dotenv()

def build_and_save_database(data_folder: str = "data", output_dir: str = "saved_db"):
    """
    Auto-scan all PDFs in data folder and build database.
    
    Args:
        data_folder: Folder containing PDF files
        output_dir: Directory to save the database files
    """
    print("="*60)
    print("🏈 BUILDING PREMIER LEAGUE RAG DATABASE FROM PDFs")
    print("="*60)
    
    # Find all PDFs
    pdf_pattern = os.path.join(data_folder, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in {data_folder}/")
        print(f"   Please add PDF files to the {data_folder}/ folder")
        return
    
    print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   - {os.path.basename(pdf)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print("\n1. Initializing Ollama and HuggingFace Embeddings...")
    
    # Initialize LLM with Ollama (LOCAL)
    try:
        llm = Ollama(
            model="gpt-oss:120b-cloud",  # Local model running on your Ollama
            base_url="http://localhost:11434",
            temperature=0,
            num_predict=4096
        )
        print("   ✓ Ollama LLM initialized (gpt-oss:120b-cloud)")  # Updated message
    except Exception as e:
        print(f"   ❌ Error connecting to Ollama: {e}")
        print("   Make sure Ollama is running with your model")
        return
    
    # Initialize embeddings with HuggingFace (CPU only)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("   ✓ HuggingFace embeddings initialized (CPU)")
    except Exception as e:
        print(f"   ❌ Error initializing embeddings: {e}")
        return
    
    # Load all documents
    print("\n2. Loading all PDFs...")
    all_documents = []
    for pdf_path in pdf_files:
        try:
            print(f"   Loading: {os.path.basename(pdf_path)}...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"   ✓ Loaded {len(documents)} pages")
        except Exception as e:
            print(f"   ✗ Error loading {pdf_path}: {e}")
    
    if not all_documents:
        print("\n❌ No documents could be loaded!")
        return
    
    print(f"\n   Total pages loaded: {len(all_documents)}")
    
    # Process documents and create embeddings
    print("\n3. Creating document embeddings...")
    processor = DocumentProcessor()
    splits, vector_store = processor.process_documents(all_documents)
    
    # Save vector store
    vector_store_path = os.path.join(output_dir, "vector_store")
    print(f"\n4. Saving vector store to {vector_store_path}...")
    processor.save_vector_store(vector_store, vector_store_path)
    
    # Build knowledge graph
    print("\n5. Building Premier League knowledge graph...")
    kg = KnowledgeGraph(llm=llm)
    kg.build_graph(splits, llm, embeddings)
    
    # Save knowledge graph
    kg_path = os.path.join(output_dir, "knowledge_graph.pkl")
    print(f"\n6. Saving knowledge graph to {kg_path}...")
    kg.save_graph(kg_path)
    
    # Print statistics
    print("\n" + "="*60)
    print("✅ DATABASE BUILD COMPLETE!")
    print("="*60)
    kg.print_graph_info()
    print(f"\nFiles saved in: {output_dir}/")
    print(f"  - vector_store/")
    print(f"  - knowledge_graph.pkl")
    print(f"  - graph_progress/ (visualizations & metadata)")
    print(f"\nProcessed {len(pdf_files)} PDF file(s)")
    print(f"Created {len(splits)} document chunks")
    print(f"Total nodes in graph: {len(kg.graph.nodes)}")
    print(f"Total edges in graph: {len(kg.graph.edges)}")
    print(f"\n🚀 You can now run the chatbot UI!")
    print("   Command: streamlit run chatbot.py")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Premier League Graph RAG Database")
    parser.add_argument('--data', type=str, default='data',
                        help='Folder containing PDF files (default: data)')
    parser.add_argument('--output', type=str, default='saved_db',
                        help='Output directory for saved database (default: saved_db)')
    
    args = parser.parse_args()
    
    build_and_save_database(args.data, args.output)
