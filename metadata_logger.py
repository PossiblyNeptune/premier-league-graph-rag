import json
import os
from datetime import datetime
from typing import Dict, List
import hashlib


class MetadataLogger:
    """Logs retrieval, traversal, connectivity, and answer quality metrics for debugging and analysis."""
    
    def __init__(self, log_dir: str = "retrieval_logs"):
        """
        Initialize metadata logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        print(f"📊 Metadata logging initialized: {log_dir}/")
    
    def log_query_session(self, 
                          query: str,
                          query_type: str,
                          retrieved_docs: List[Dict],
                          traversal_path: List[int],
                          traversal_decisions: List[Dict],
                          filtered_content: Dict[int, str],
                          final_answer: str,
                          response_time: float,
                          answer_confidence: float = None,
                          is_complete: bool = None,
                          missing_elements: List[str] = None) -> str:
        """
        Logs a complete query session with all debugging info.
        
        Args:
            query: The original query
            query_type: Classified query type (manager, player, tactics, etc.)
            retrieved_docs: Initial vector store retrieval results
            traversal_path: Path through knowledge graph (list of node IDs)
            traversal_decisions: Individual node selection decisions
            filtered_content: Content from each node visited
            final_answer: Generated answer text
            response_time: Total response time in seconds
            answer_confidence: Confidence score (0-1)
            is_complete: Whether answer was complete
            missing_elements: Elements missing from answer
            
        Returns:
            session_id for reference
        """
        session_id = hashlib.md5(
            (query + datetime.now().isoformat()).encode()
        ).hexdigest()[:12]
        
        connectivity_metrics = self._calculate_connectivity_metrics(traversal_path)
        
        log_entry = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_hash': hashlib.md5(query.encode()).hexdigest(),
            'query_type': query_type,
            
            # ============ RETRIEVAL PHASE ============
            'initial_retrieval': {
                'num_docs_retrieved': len(retrieved_docs),
                'doc_sources': [
                    doc.metadata.get('source', 'unknown')
                    for doc in retrieved_docs
                ] if retrieved_docs else [],
                'retrieval_diversity': self._calculate_retrieval_diversity(retrieved_docs),
            },            
            
            # ============ GRAPH TRAVERSAL PHASE ============
            'graph_traversal': {
                'nodes_visited': traversal_path,
                'num_nodes': len(traversal_path),
                'traversal_depth': len(traversal_path),
                'avg_edge_weight': self._calculate_avg_edge_weight(traversal_decisions),
                'max_edge_weight': self._calculate_max_edge_weight(traversal_decisions),
                'min_edge_weight': self._calculate_min_edge_weight(traversal_decisions),
            },
            
            # ============ GRAPH CONNECTIVITY ============
            'graph_connectivity': connectivity_metrics,
            
            # ============ CONTEXT QUALITY ============
            'context_quality': {
                'total_context_length': sum(len(content) for content in filtered_content.values()),
                'avg_chunk_length': (
                    sum(len(content) for content in filtered_content.values()) / len(filtered_content)
                    if filtered_content else 0
                ),
                'num_chunks': len(filtered_content),
                'context_compression': self._calculate_compression_ratio(
                    len(retrieved_docs), len(filtered_content)
                ),
            },
            
            # ============ ANSWER QUALITY ============
            'response_quality': {
                'answer_length': len(final_answer),
                'answer_preview': final_answer[:200],
                'confidence_score': answer_confidence,
                'is_complete': is_complete,
                'missing_elements': missing_elements or [],
                'response_time_seconds': response_time,
                'tokens_generated': len(final_answer.split()),
                'answer_coherence': self._estimate_coherence(final_answer),
            },
            
            # ============ EFFICIENCY METRICS ============
            'efficiency_metrics': {
                'retrieval_to_traversal_ratio': (
                    len(traversal_path) / max(len(retrieved_docs), 1)
                ),
                'context_utilization': (
                    sum(len(content) for content in filtered_content.values()) / 4000
                ),
                'response_time_per_node': response_time / max(len(traversal_path), 1),
                'response_time_per_doc': response_time / max(len(retrieved_docs), 1),
            }
        }
        
        log_file = os.path.join(self.log_dir, f"{session_id}.json")
        try:
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
            print(f"📊 Session logged: {session_id}")
        except Exception as e:
            print(f"⚠️  Could not save log: {e}")
        
        return session_id
    
    def log_traversal_decision(self,
                               current_node: int,
                               target_node: int,
                               edge_weight: float,
                               shared_concepts: List[str],
                               concept_relevance: float,
                               reason_selected: str,
                               accumulated_context_length: int,
                               traversal_depth: int):
        """Logs individual traversal decisions for debugging."""
        decision = {
            'current_node': current_node,
            'target_node': target_node,
            'edge_weight': edge_weight,
            'shared_concepts': shared_concepts,
            'num_shared_concepts': len(shared_concepts),
            'concept_relevance': concept_relevance,
            'reason_selected': reason_selected,
            'context_length_at_step': accumulated_context_length,
            'traversal_depth': traversal_depth,
            'timestamp': datetime.now().isoformat()
        }
        return decision
    
    # ============ HELPER METHODS ============
    
    @staticmethod
    def _calculate_avg_edge_weight(traversal_decisions: List[Dict]) -> float:
        """Calculate average edge weight from traversal decisions."""
        if not traversal_decisions:
            return 0.0
        
        weights = [d.get('edge_weight', 0) for d in traversal_decisions]
        valid_weights = [w for w in weights if w is not None and isinstance(w, (int, float))]
        
        return round(sum(valid_weights) / len(valid_weights), 4) if valid_weights else 0.0
    
    @staticmethod
    def _calculate_max_edge_weight(traversal_decisions: List[Dict]) -> float:
        """Calculate max edge weight."""
        if not traversal_decisions:
            return 0.0
        
        weights = [d.get('edge_weight', 0) for d in traversal_decisions]
        valid_weights = [w for w in weights if w is not None and isinstance(w, (int, float))]
        
        return round(max(valid_weights), 4) if valid_weights else 0.0
    
    @staticmethod
    def _calculate_min_edge_weight(traversal_decisions: List[Dict]) -> float:
        """Calculate min edge weight."""
        if not traversal_decisions:
            return 0.0
        
        weights = [d.get('edge_weight', 0) for d in traversal_decisions]
        valid_weights = [w for w in weights if w is not None and isinstance(w, (int, float))]
        
        return round(min(valid_weights), 4) if valid_weights else 0.0
    
    @staticmethod
    def _calculate_retrieval_diversity(retrieved_docs: List[Dict]) -> float:
        """Measure diversity of retrieved documents."""
        if len(retrieved_docs) < 2:
            return 0.0
        
        sources = set(doc.metadata.get('source', 'unknown') for doc in retrieved_docs)
        return round(len(sources) / len(retrieved_docs), 4)
    
    @staticmethod
    def _calculate_compression_ratio(num_initial: int, num_final: int) -> float:
        """How much traversal compressed the context."""
        if num_initial == 0:
            return 0.0
        
        return round((1 - (num_final / num_initial)) * 100, 2)  # Percentage reduction
    
    @staticmethod
    def _calculate_connectivity_metrics(traversal_path: List[int]) -> Dict:
        """Analyze path connectivity patterns."""
        if len(traversal_path) < 2:
            return {
                'path_length': len(traversal_path),
                'is_linear': True,
                'has_backtracking': False,
                'avg_step_distance': 0.0,
            }
        
        jumps = 0
        total_distance = 0
        for i in range(len(traversal_path) - 1):
            distance = abs(traversal_path[i+1] - traversal_path[i])
            total_distance += distance
            if distance > 1:
                jumps += 1
        
        return {
            'path_length': len(traversal_path),
            'is_linear': jumps == 0,
            'num_jumps': jumps,
            'avg_step_distance': round(total_distance / (len(traversal_path) - 1), 2),
            'has_backtracking': any(traversal_path[i] > traversal_path[i+1] for i in range(len(traversal_path)-1)),
        }
    
    @staticmethod
    def _estimate_coherence(answer_text: str) -> float:
        """Estimate answer coherence (0-1)."""
        if not answer_text or len(answer_text) < 50:
            return 0.3
        
        score = 0.7
        
        if any(phrase in answer_text.lower() for phrase in ['because', 'however', 'therefore']):
            score += 0.1
        
        if answer_text.count('.') < 2:
            score -= 0.2
        
        if any(word in answer_text.lower() for word in ['specifically', 'example', 'achieved', 'won']):
            score += 0.1
        
        return round(min(max(score, 0.0), 1.0), 2)
    
    @staticmethod
    def _count_query_types(sessions: List[Dict]) -> Dict:
        """Count distribution of query types."""
        counts = {}
        for session in sessions:
            query_type = session.get('query_type', 'unknown')
            counts[query_type] = counts.get(query_type, 0) + 1
        
        return counts
