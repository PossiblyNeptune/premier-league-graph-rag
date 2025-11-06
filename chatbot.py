"""
Streamlit Chatbot UI for Premier League Football Knowledge
Enhanced debugging, metadata display, and general football theming.
Based on **The Mixer** (Michael Cox - Tactics) and **The Club** (Robinson & Clegg - Business).
Uses LOCAL Ollama.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
import pickle
import traceback

from create_embeddings import DocumentProcessor
from create_database import KnowledgeGraph
from graph_rag import QueryEngine
from metadata_logger import MetadataLogger

load_dotenv()


@st.cache_resource
def load_chatbot(db_path: str = "saved_db"):
    """Load chatbot with pre-built database (cached)."""
    
    llm = Ollama(
        model="gpt-oss:120b-cloud",  # Change from "mistral"
        base_url="http://localhost:11434",
        temperature=0,
        num_predict=4096
    )
    
    # Load vector store
    processor = DocumentProcessor()
    vector_store_path = os.path.join(db_path, "vector_store")
    vector_store = processor.load_vector_store(vector_store_path)
    
    # Load knowledge graph
    knowledge_graph = KnowledgeGraph()
    kg_path = os.path.join(db_path, "knowledge_graph.pkl")
    knowledge_graph.load_graph(kg_path)

    logger = MetadataLogger(graph=knowledge_graph.graph)
    
    # Initialize query engine
    query_engine = QueryEngine(vector_store, knowledge_graph, llm)
    
    return query_engine, vector_store, knowledge_graph


@st.cache_resource
def load_pdf_mapping():
    """Create mapping of document content to PDF source."""
    pdf_mapping = {}
    data_dir = "data"
    
    if os.path.exists(data_dir):
        for pdf_file in os.listdir(data_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(data_dir, pdf_file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    for doc in docs:
                        content_start = doc.page_content[:100]
                        pdf_mapping[content_start] = {
                            'filename': pdf_file,
                            'page': doc.metadata.get('page', 'Unknown')
                        }
                except:
                    pass
    
    return pdf_mapping


def get_pdf_source(content: str, pdf_mapping: dict) -> dict:
    """Find which PDF a chunk came from."""
    content_start = content[:100]
    
    for key, value in pdf_mapping.items():
        if key in content or content_start.startswith(key[:50]):
            return value
    
    return {'filename': 'Unknown', 'page': 'Unknown'}


def main():
    st.set_page_config(
        page_title="⚽ Premier League Knowledge Bot",
        page_icon="⚽",
        layout="wide"
    )
    
    # Custom CSS - Premier League Theme
    st.markdown("""
        <style>
        .chunk-box {
            background-color: #f4f4f4;
            border-left: 4px solid #2e7d32;
            padding: 12px;
            margin: 8px 0;
            border-radius: 5px;
            font-size: 13px;
        }
        .pdf-label {
            background-color: #c8e6c9;
            color: #1b5e20;
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
            display: inline-block;
            margin-right: 8px;
        }
        .traversal-path {
            background-color: #fff9c4;
            border-left: 4px solid #f57f17;
            padding: 12px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .node-info {
            background-color: #e0f2f1;
            border: 1px solid #00897b;
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            font-size: 12px;
        }
        .response-container {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .query-type-badge {
            display: inline-block;
            padding: 4px 12px;
            background-color: #1565c0;
            color: white;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚽ Premier League Knowledge Bot")
    st.markdown("""
    Explore Premier League history, business, tactics, players, managers, and teams. 
    Powered by **The Mixer** (Michael Cox - Tactics) and **The Club** (Robinson & Clegg - Business)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📖 About This Bot")
        st.markdown("""
        This chatbot provides insights into:
        - **Tactics & Strategy** - Formation evolution, pressing systems, tactical philosophy
        - **Club History** - How Premier League clubs developed, ownership, culture
        - **Players** - Legends, their impact, playing style
        - **Managers** - Coaching philosophies, achievements, influence
        - **Business & Money** - Club finances, transfers, foreign investment
        - **Team Performance** - Historic seasons, achievements
        
        ### How to use:
        1. Run `python build_database.py` to build the knowledge base
        2. Ask any question about Premier League football
        3. Get context-aware answers with sources
        
        ### Example Questions:
        - "What was Arsene Wenger's impact on Arsenal?"
        - "How did Manchester United dominate the 90s?"
        - "Describe the 4-3-3 formation"
        - "Who were key players in Liverpool's success?"
        - "How did pressing tactics change English football?"
        - "Compare different managerial approaches"
        - "What changed when Wenger focused on diet?"
        """)
        
        st.divider()
        
        # Database info
        db_path = "saved_db"
        if os.path.exists(db_path):
            st.success("✅ Knowledge Base Loaded")
            kg_path = os.path.join(db_path, "knowledge_graph.pkl")
            if os.path.exists(kg_path):
                with open(kg_path, 'rb') as f:
                    graph_data = pickle.load(f)
                    num_nodes = len(graph_data['graph'].nodes)
                    num_edges = len(graph_data['graph'].edges)
                    st.metric("Knowledge Concepts (Nodes)", num_nodes)
                    st.metric("Connections (Edges)", num_edges)
                    st.metric("Graph Density", f"{len(graph_data['graph'].edges) / max(num_nodes * (num_nodes - 1) / 2, 1):.3f}")
        else:
            st.error("❌ Knowledge Base Not Found")
            st.info("Run: `python build_database.py`")
        
        if st.button("🔄 Reload Database"):
            st.cache_resource.clear()
            st.rerun()
        
        # Debug toggle
        debug_mode = st.checkbox("🔍 Show Retrieval Details", value=False)
        
        st.divider()

        st.subheader("📊 Graph Connectivity")

        # Display connectivity metrics if available
        if os.path.exists("graph_visualizations/connectivity_report.json"):
            import json
            with open("graph_visualizations/connectivity_report.json", 'r') as f:
                connectivity = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Is Connected", "✅ Yes" if connectivity['is_connected'] else "❌ No")
                st.metric("Components", connectivity['num_components'])
            with col2:
                st.metric("Largest Comp.", f"{connectivity['largest_component_size']} nodes")
                st.metric("Isolated Nodes", connectivity['isolated_nodes'])
            
            st.divider()
            
            if st.button("📂 Open Graph Dashboard"):
                st.info("Open `graph_visualizations/dashboard.html` in your browser")
        st.caption("Made with ⚽ and knowledge graphs | Powered by Local Ollama")
    
    # Check if database exists
    if not os.path.exists("saved_db"):
        st.error("⚠️ Knowledge base not found!")
        st.info("Please run the following command first:")
        st.code("python build_database.py", language="bash")
        return
    
    # Load chatbot
    try:
        with st.spinner("Loading Premier League knowledge..."):
            query_engine, vector_store, knowledge_graph = load_chatbot()
            pdf_mapping = load_pdf_mapping()
        st.success("✅ Bot ready to answer your questions!")
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        if debug_mode:
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info if available
            if debug_mode and "debug_info" in message:
                with st.expander("📊 Retrieval Details"):
                    debug_info = message["debug_info"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Graph Nodes Visited", len(debug_info.get('traversal_path', [])))
                    with col2:
                        st.metric("Total Context Length", f"{len(debug_info.get('context', '')) // 100} x100 chars")
                    with col3:
                        st.metric("Query Category", debug_info.get('query_type', 'General').title())
                    
                    # Traversal path
                    if debug_info.get('traversal_path'):
                        st.write("**Knowledge Graph Path:**")
                        st.code(f"Nodes visited: {debug_info['traversal_path']}", language="text")
                    
                    # Chunks used
                    if debug_info.get('chunks'):
                        st.write("**Sources Referenced:**")
                        for i, chunk_info in enumerate(debug_info['chunks'], 1):
                            with st.expander(f"📄 Reference {i} - {chunk_info['source']} (Page {chunk_info['page']})"):
                                st.markdown(f"<div class='chunk-box'>{chunk_info['content'][:500]}...</div>", 
                                          unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about Premier League football, players, managers, tactics..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    response, traversal_path, filtered_content = query_engine.query(prompt)
                    
                    # Extract response text
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                    
                    st.markdown(response_text)
                    
                    # Prepare debug info
                    debug_info = {
                        'traversal_path': traversal_path,
                        'query_type': query_engine._classify_query(prompt),
                        'context': '',
                        'chunks': []
                    }
                    
                    # Extract chunk information
                    for node_id in traversal_path:
                        if node_id in filtered_content:
                            content = filtered_content[node_id]
                            source_info = get_pdf_source(content, pdf_mapping)
                            
                            debug_info['chunks'].append({
                                'node_id': node_id,
                                'source': source_info['filename'],
                                'page': source_info['page'],
                                'content': content
                            })
                    
                    # Show debug info in expander
                    if debug_mode:
                        with st.expander("🔍 Retrieval Analysis"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Nodes Traversed", len(traversal_path))
                            with col2:
                                st.metric("Chunks Retrieved", len(debug_info['chunks']))
                            with col3:
                                st.metric("Query Type", debug_info['query_type'].title())
                            with col4:
                                st.metric("Total Edges Used", len(traversal_path) - 1)
                            
                            # Sources breakdown
                            st.write("**Sources Used:**")
                            source_counts = {}
                            for chunk in debug_info['chunks']:
                                source = chunk['source']
                                source_counts[source] = source_counts.get(source, 0) + 1
                            
                            source_cols = st.columns(len(source_counts)) if source_counts else [None]
                            for col, (source, count) in zip(source_cols, source_counts.items()):
                                if col:
                                    with col:
                                        st.metric(source, f"{count} ref")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "debug_info": debug_info
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    
                    if debug_mode:
                        with st.expander("🔧 Error Details"):
                            st.code(traceback.format_exc(), language="python")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "error": traceback.format_exc()
                    })
    
    # Footer with chat controls
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🔴 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col3:
        if st.button("💾 Save Chat"):
            st.info("Chat history is automatically saved in the session!")


if __name__ == "__main__":
    main()
