import networkx as nx
import pickle
import os
import time
import nltk
import spacy
from spacy.cli import download
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv
import json

from graph_visualizer import GraphVisualizer
from football_tactics_preprocessor import FootballTacticsPreprocessor

load_dotenv()

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


class KnowledgeGraph:
    def __init__(self, llm=None, edges_threshold=0.7, 
                 save_progress=True, progress_dir="graph_progress"):
        """
        Initializes the KnowledgeGraph with domain-optimized settings.

        Args:
        - llm: An instance of a large language model (optional, can be set later).
        - edges_threshold (float): The threshold for adding edges based on similarity. 
        - save_progress (bool): Whether to save progress visualizations. Default is True.
        - progress_dir (str): Directory to save progress visualizations.
        """
        self.graph = nx.Graph()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = edges_threshold
        self.llm = llm
        self.save_progress = save_progress
        self.progress_dir = progress_dir
        self.preprocessor = FootballTacticsPreprocessor()
        self.extraction_metadata = []
        
        if self.save_progress:
            os.makedirs(self.progress_dir, exist_ok=True)
            print(f"📊 Progress visualizations will be saved to: {self.progress_dir}/")
        
        print("🏈 KnowledgeGraph initialized with football tactics optimization")

    def _load_spacy_model(self):
        """
        Loads the spaCy NLP model, downloading it if necessary.

        Returns:
        - spacy.Language: An instance of a spaCy NLP model.
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content):
        """
        Extracts concepts using spaCy + football preprocessing (NO LLM needed).
        Fast, reliable, and effective for knowledge graph construction.
        """
        if content in self.concept_cache:
            return self.concept_cache[content]

        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

        tactical_entities = self.preprocessor.extract_tactical_entities(content)
        
        tactical_concepts = (
            tactical_entities.get('formations', []) +
            tactical_entities.get('tactical_roles', []) +
            tactical_entities.get('defensive_concepts', []) +
            tactical_entities.get('offensive_concepts', [])
        )

        # Combine all concepts (NO LLM - fast and reliable)
        all_concepts = list(set(named_entities + tactical_concepts))[:15]
        
        self.extraction_metadata.append({
            'content_hash': hash(content[:100]),
            'named_entities_count': len(named_entities),
            'tactical_concepts_count': len(tactical_concepts),
            'total_concepts': len(all_concepts),
            'extraction_success': len(all_concepts) > 0
        })

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _compute_similarities(self, embeddings):
        """
        Computes the cosine similarity matrix for the embeddings.

        Args:
        - embeddings (list or numpy.ndarray): A list or array of embeddings.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the embeddings.
        """
        return cosine_similarity(embeddings)

    def build_graph(self, splits, llm, embedding_model):
        """
        Builds the knowledge graph with football tactics optimizations.

        Args:
        - splits (list): A list of document splits.
        - llm: An instance of a large language model.
        - embedding_model: An instance of an embedding model.

        Returns:
        - None
        """
        print("\n" + "="*60)
        print("🏈 Building Football Tactics Knowledge Graph")
        print("="*60)
        
        self.llm = llm
        
        print("\nStep 1: Adding nodes to graph...")
        self._add_nodes(splits)
        
        print("\nStep 2: Creating embeddings...")
        embeddings = self._create_embeddings(splits, embedding_model)
        
        print("\nStep 3: Extracting football tactical concepts and entities...")
        self._extract_concepts(splits)
        
        print("\nStep 4: Adding edges based on similarity and shared concepts...")
        self._add_edges(embeddings)
        
        self._save_extraction_metadata()
        
        print("\n" + "="*60)
        print(f"✅ Knowledge Graph Built Successfully!")
        print(f"    Total Nodes: {len(self.graph.nodes)}")
        print(f"    Total Edges: {len(self.graph.edges)}")
        print(f"    Extraction Success Rate: {self._get_extraction_success_rate():.1%}")
        print("="*60)

        print("\n📊 Generating graph visualizations...")   
        visualizer = GraphVisualizer(self.graph, output_dir="graph_visualizations")
        visualizer.generate_all_visualizations()

    def _add_nodes(self, splits):
        """
        Adds nodes to the graph from the document splits with enhanced metadata.

        Args:
        - splits (list): A list of document splits.
        """
        for i, split in enumerate(splits):
            self.graph.add_node(
                i,
                content=split.page_content,
                metadata=split.metadata,
                source_file=split.metadata.get('source_file', 'unknown'),
                page_number=split.metadata.get('page_number', 'unknown'),
                chunk_id=split.metadata.get('chunk_id', i)
            )
        print(f"✓ Added {len(splits)} nodes to the graph")

    def _create_embeddings(self, splits, embedding_model):
        """
        Creates embeddings for the document splits using the embedding model.

        Args:
        - splits (list): A list of document splits.
        - embedding_model: An instance of an embedding model.

        Returns:
        - list: A list of embeddings for the document splits.
        """
        texts = [split.page_content for split in splits]
        print(f"Creating embeddings for {len(texts)} text chunks...")
        
        # Batch embeddings
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print(f"  Processing embedding batch {batch_num}/{total_batches}...")
            
            try:
                batch_embeddings = embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"  ⚠️  Error creating embeddings: {e}")
                import numpy as np
                all_embeddings.extend([np.zeros(384).tolist() for _ in batch])
        
        print(f"✓ Created {len(all_embeddings)} embeddings")
        return all_embeddings

    def _extract_concepts(self, splits):
        """
        Extracts concepts for all document splits.

        Args:
        - splits (list): A list of document splits.
        """
        print(f"Processing {len(splits)} chunks for concept extraction...")
        
        for i, split in enumerate(tqdm(splits, desc="Extracting concepts")):
            try:
                concepts = self._extract_concepts_and_entities(split.page_content)
                self.graph.nodes[i]['concepts'] = concepts
            except Exception as e:
                print(f"Error processing node {i}: {e}")
                self.graph.nodes[i]['concepts'] = []

    def _add_edges(self, embeddings):
        """
        Adds edges to the graph based on similarity and shared concepts.

        Args:
        - embeddings (list or numpy.ndarray): A list or array of embeddings for the document splits.
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        edges_added = 0
        
        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                
                # Use LOWER threshold (0.7) for football domain specificity
                if similarity_score > self.edges_threshold:
                    
                    concepts1 = set(self.graph.nodes[node1].get('concepts', []))
                    concepts2 = set(self.graph.nodes[node2].get('concepts', []))
                    shared_concepts = concepts1 & concepts2
                    
                    edge_weight = self._calculate_edge_weight(
                        node1, node2,
                        similarity_score,
                        shared_concepts
                    )
                    
                    confidence_score = min(similarity_score, 1.0)
                    
                    self.graph.add_edge(
                        node1, node2,
                        weight=edge_weight,
                        similarity=similarity_score,
                        shared_concepts=list(shared_concepts),
                        confidence=confidence_score,
                        edge_type='semantic'
                    )
                    edges_added += 1
        
        print(f"✓ Added {edges_added} edges to the graph")

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        Calculates the weight of an edge based on similarity score and shared concepts.

        Args:
        - node1 (int): The first node.
        - node2 (int): The second node.
        - similarity_score (float): The similarity score between the nodes.
        - shared_concepts (set): The set of shared concepts between the nodes.
        - alpha (float, optional): The weight of the similarity score. Default is 0.7.
        - beta (float, optional): The weight of the shared concepts. Default is 0.3.

        Returns:
        - float: The calculated weight of the edge.
        """
        max_possible_shared = min(
            len(self.graph.nodes[node1].get('concepts', [])),
            len(self.graph.nodes[node2].get('concepts', []))
        )
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        
        return alpha * similarity_score + beta * normalized_shared_concepts

    def save_graph(self, path: str = "knowledge_graph.pkl"):
        """
        Saves the knowledge graph to disk with metadata.

        Args:
        - path (str): The file path where the graph will be saved.
        """
        print(f"Saving knowledge graph to {path}...")
        graph_data = {
            'graph': self.graph,
            'concept_cache': self.concept_cache,
            'edges_threshold': self.edges_threshold,
            'extraction_metadata': self.extraction_metadata,
            'timestamp': time.time()
        }
        with open(path, 'wb') as f:
            pickle.dump(graph_data, f)
        print("✓ Knowledge graph saved successfully.")

    def load_graph(self, path: str = "knowledge_graph.pkl"):
        """
        Loads the knowledge graph from disk.

        Args:
        - path (str): The file path where the graph is saved.
        """
        print(f"Loading knowledge graph from {path}...")
        with open(path, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.graph = graph_data['graph']
        self.concept_cache = graph_data['concept_cache']
        self.edges_threshold = graph_data['edges_threshold']
        self.extraction_metadata = graph_data.get('extraction_metadata', [])
        print(f"✓ Knowledge graph loaded successfully.")
        print(f"  Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)}")

    def get_graph_statistics(self) -> dict:
        """
        Returns statistics about the knowledge graph.

        Returns:
        - dict: A dictionary containing graph statistics.
        """
        stats = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if len(self.graph.nodes) > 0 else 0,
            'density': nx.density(self.graph) if len(self.graph.nodes) > 0 else 0,
            'is_connected': nx.is_connected(self.graph) if len(self.graph.nodes) > 0 else False,
            'num_connected_components': nx.number_connected_components(self.graph) if len(self.graph.nodes) > 0 else 0
        }
        return stats

    def print_graph_info(self):
        """
        Prints detailed information about the knowledge graph.
        """
        stats = self.get_graph_statistics()
        
        print("\n" + "="*60)
        print("🏈 Football Tactics Knowledge Graph Statistics")
        print("="*60)
        print(f"Number of Nodes: {stats['num_nodes']}")
        print(f"Number of Edges: {stats['num_edges']}")
        print(f"Average Degree: {stats['average_degree']:.2f}")
        print(f"Graph Density: {stats['density']:.4f}")
        print(f"Is Connected: {stats['is_connected']}")
        print(f"Number of Connected Components: {stats['num_connected_components']}")
        print("="*60)
    
    def _save_extraction_metadata(self):
        """Save extraction metadata to JSON for analysis."""
        if not self.extraction_metadata:
            return
        
        metadata_file = os.path.join(self.progress_dir, "extraction_metadata.json")
        
        total_extractions = len(self.extraction_metadata)
        successful = sum(1 for m in self.extraction_metadata if m['extraction_success'])
        
        summary = {
            'total_extractions': total_extractions,
            'successful_extractions': successful,
            'success_rate': successful / total_extractions if total_extractions > 0 else 0,
            'avg_concepts_per_node': sum(m['total_concepts'] for m in self.extraction_metadata) / total_extractions if total_extractions > 0 else 0,
        }
        
        with open(metadata_file, 'w') as f:
            json.dump({
                'summary': summary,
                'metadata': self.extraction_metadata
            }, f, indent=2)
        
        print(f"✓ Extraction metadata saved to {metadata_file}")
    
    def _get_extraction_success_rate(self) -> float:
        """Get extraction success rate."""
        if not self.extraction_metadata:
            return 0
        
        successful = sum(1 for m in self.extraction_metadata if m['extraction_success'])
        return successful / len(self.extraction_metadata)
