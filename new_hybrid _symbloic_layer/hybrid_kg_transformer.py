import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict
import time
import json

@dataclass
class KnowledgeNode:
    """Enhanced knowledge node with both neural and symbolic representations"""
    entity_id: str
    neural_embedding: torch.Tensor
    symbolic_data: Dict[str, Any]
    concept_type: str
    confidence: float
    usage_count: float
    timestamp: float
    connections: List[str]

class HybridKnowledgeManager:
    """
    Combines neural sparse storage with symbolic graph reasoning
    Best of both worlds: efficiency + interpretability
    """
    
    def __init__(
        self, 
        hidden_dim: int = 768,
        max_neural_nodes: int = 1000,
        symbolic_graph_limit: int = 5000
    ):
        self.hidden_dim = hidden_dim
        self.max_neural_nodes = max_neural_nodes
        self.symbolic_graph_limit = symbolic_graph_limit
        
        # Neural storage (GPU-efficient)
        self.neural_embeddings = torch.zeros(max_neural_nodes, hidden_dim)
        self.neural_active = torch.zeros(max_neural_nodes, dtype=torch.bool)
        self.neural_usage = torch.zeros(max_neural_nodes)
        self.current_neural_nodes = 0
        
        # Symbolic storage (CPU-efficient)
        self.symbolic_graph = nx.DiGraph()
        self.entity_to_neural_idx = {}  # Map entities to neural indices
        self.neural_idx_to_entity = {}  # Reverse mapping
        
        # Hybrid indices
        self.concept_hierarchy = defaultdict(set)
        self.entity_metadata = {}
        
    def add_knowledge(
        self, 
        entity: str, 
        embedding: torch.Tensor, 
        concept_type: str = "general",
        symbolic_data: Optional[Dict] = None,
        confidence: float = 1.0
    ) -> int:
        """Add knowledge with both neural and symbolic components"""
        
        # Add to neural storage
        neural_idx = -1
        if self.current_neural_nodes < self.max_neural_nodes:
            neural_idx = self.current_neural_nodes
            self.neural_embeddings[neural_idx] = embedding.detach()
            self.neural_active[neural_idx] = True
            self.current_neural_nodes += 1
            
            # Create mappings
            self.entity_to_neural_idx[entity] = neural_idx
            self.neural_idx_to_entity[neural_idx] = entity
        
        # Add to symbolic storage
        node_data = {
            'concept_type': concept_type,
            'confidence': confidence,
            'timestamp': time.time(),
            'neural_idx': neural_idx,
            'symbolic_data': symbolic_data or {}
        }
        
        self.symbolic_graph.add_node(entity, **node_data)
        self.concept_hierarchy[concept_type].add(entity)
        self.entity_metadata[entity] = node_data
        
        return neural_idx
    
    def add_relation(self, source: str, target: str, relation_type: str, strength: float = 1.0):
        """Add symbolic relation between entities"""
        if source in self.symbolic_graph and target in self.symbolic_graph:
            self.symbolic_graph.add_edge(
                source, target,
                relation_type=relation_type,
                strength=strength,
                timestamp=time.time()
            )
    
    def retrieve_hybrid(
        self, 
        query_embedding: torch.Tensor, 
        top_k_neural: int = 5,
        max_symbolic_hops: int = 2
    ) -> Dict[str, Any]:
        """Retrieve using both neural similarity and symbolic reasoning"""
        
        results = {
            'neural_matches': [],
            'symbolic_context': [],
            'hybrid_score': 0.0
        }
        
        # Neural retrieval
        if self.current_neural_nodes > 0:
            active_embeddings = self.neural_embeddings[self.neural_active]
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(0), 
                active_embeddings, 
                dim=-1
            )
            
            top_k = min(top_k_neural, similarities.shape[0])
            top_similarities, top_indices = torch.topk(similarities, top_k)
            
            for i, (sim, idx) in enumerate(zip(top_similarities, top_indices)):
                neural_idx = torch.where(self.neural_active)[0][idx].item()
                entity = self.neural_idx_to_entity.get(neural_idx)
                if entity:
                    results['neural_matches'].append({
                        'entity': entity,
                        'similarity': sim.item(),
                        'neural_idx': neural_idx,
                        'metadata': self.entity_metadata.get(entity, {})
                    })
        
        # Symbolic reasoning from neural matches
        for match in results['neural_matches'][:3]:  # Top 3 for symbolic expansion
            entity = match['entity']
            symbolic_context = self._get_symbolic_context(entity, max_symbolic_hops)
            results['symbolic_context'].extend(symbolic_context)
        
        # Compute hybrid score
        neural_score = sum(m['similarity'] for m in results['neural_matches'][:3])
        symbolic_score = len(results['symbolic_context']) * 0.1
        results['hybrid_score'] = neural_score + symbolic_score
        
        return results
    
    def _get_symbolic_context(self, entity: str, max_hops: int) -> List[Dict]:
        """Get symbolic context through graph traversal"""
        context = []
        if entity not in self.symbolic_graph:
            return context
        
        # BFS traversal for context
        visited = {entity}
        current_level = [entity]
        
        for hop in range(max_hops):
            next_level = []
            for node in current_level:
                # Outgoing edges
                for neighbor in self.symbolic_graph.successors(node):
                    if neighbor not in visited:
                        edge_data = self.symbolic_graph[node][neighbor]
                        context.append({
                            'source': node,
                            'target': neighbor,
                            'relation': edge_data.get('relation_type', 'related'),
                            'strength': edge_data.get('strength', 1.0),
                            'hop_distance': hop + 1
                        })
                        visited.add(neighbor)
                        next_level.append(neighbor)
                
                # Incoming edges
                for predecessor in self.symbolic_graph.predecessors(node):
                    if predecessor not in visited:
                        edge_data = self.symbolic_graph[predecessor][node]
                        context.append({
                            'source': predecessor,
                            'target': node,
                            'relation': edge_data.get('relation_type', 'related'),
                            'strength': edge_data.get('strength', 1.0),
                            'hop_distance': hop + 1
                        })
                        visited.add(predecessor)
                        next_level.append(predecessor)
            
            current_level = next_level
            if not current_level:
                break
        
        return context

class CreativeReasoningLayer:
    """Enhanced creative layer with both neural and symbolic creativity"""
    
    def __init__(self, exploration_factor: float = 0.3):
        self.exploration_factor = exploration_factor
        self.creative_patterns = []
        self.reasoning_cache = {}
    
    def generate_creative_paths(
        self,
        start_embedding: torch.Tensor,
        end_embedding: torch.Tensor,
        knowledge_manager: HybridKnowledgeManager,
        num_paths: int = 3
    ) -> List[Dict]:
        """Generate creative paths using both neural interpolation and symbolic reasoning"""
        
        creative_paths = []
        
        # Get hybrid context for both start and end
        start_context = knowledge_manager.retrieve_hybrid(start_embedding, top_k_neural=3)
        end_context = knowledge_manager.retrieve_hybrid(end_embedding, top_k_neural=3)
        
        for i in range(num_paths):
            path_data = {
                'neural_path': [],
                'symbolic_chain': [],
                'creativity_score': 0.0,
                'method': 'hybrid'
            }
            
            # Neural creative interpolation
            neural_path = self._create_neural_path(start_embedding, end_embedding, i)
            path_data['neural_path'] = neural_path
            
            # Symbolic reasoning chain
            if start_context['neural_matches'] and end_context['neural_matches']:
                symbolic_chain = self._create_symbolic_chain(
                    start_context, end_context, knowledge_manager
                )
                path_data['symbolic_chain'] = symbolic_chain
            
            # Score creativity
            path_data['creativity_score'] = self._score_creativity(path_data)
            creative_paths.append(path_data)
        
        return sorted(creative_paths, key=lambda x: x['creativity_score'], reverse=True)
    
    def _create_neural_path(self, start: torch.Tensor, end: torch.Tensor, variant: int) -> List[torch.Tensor]:
        """Create neural interpolation path with creative variations"""
        path = [start.clone()]
        
        steps = 3 + variant  # Different path lengths
        for step in range(1, steps):
            alpha = step / steps
            
            # Add creative noise based on variant
            creative_noise = torch.randn_like(start) * self.exploration_factor * (0.5 - abs(alpha - 0.5))
            
            interpolated = (1 - alpha) * start + alpha * end + creative_noise
            path.append(interpolated)
        
        path.append(end.clone())
        return path
    
    def _create_symbolic_chain(
        self, 
        start_context: Dict, 
        end_context: Dict, 
        knowledge_manager: HybridKnowledgeManager
    ) -> List[Dict]:
        """Create symbolic reasoning chain connecting start to end"""
        chain = []
        
        # Find conceptual bridges
        start_concepts = {m['metadata'].get('concept_type', 'general') 
                         for m in start_context['neural_matches']}
        end_concepts = {m['metadata'].get('concept_type', 'general') 
                       for m in end_context['neural_matches']}
        
        # Create reasoning steps
        for start_concept in start_concepts:
            for end_concept in end_concepts:
                if start_concept != end_concept:
                    # Find intermediate concepts
                    intermediate = self._find_concept_bridge(
                        start_concept, end_concept, knowledge_manager
                    )
                    
                    chain.append({
                        'from_concept': start_concept,
                        'to_concept': end_concept,
                        'intermediate': intermediate,
                        'reasoning': f"Transform {start_concept} through {intermediate} to {end_concept}"
                    })
        
        return chain
    
    def _find_concept_bridge(self, start: str, end: str, knowledge_manager: HybridKnowledgeManager) -> str:
        """Find conceptual bridge between two concepts"""
        # Simple heuristic - could be enhanced with more sophisticated reasoning
        all_concepts = list(knowledge_manager.concept_hierarchy.keys())
        
        # Find concept that's "between" start and end alphabetically (simple heuristic)
        bridges = [c for c in all_concepts if start < c < end or end < c < start]
        return bridges[0] if bridges else "transition"
    
    def _score_creativity(self, path_data: Dict) -> float:
        """Score the creativity of a generated path"""
        base_score = 0.5
        
        # Neural path diversity
        if len(path_data['neural_path']) > 3:
            base_score += 0.2
        
        # Symbolic reasoning complexity
        if len(path_data['symbolic_chain']) > 1:
            base_score += 0.3
        
        # Novelty bonus
        path_signature = f"{len(path_data['neural_path'])}_{len(path_data['symbolic_chain'])}"
        if path_signature not in self.reasoning_cache:
            base_score += 0.2
            self.reasoning_cache[path_signature] = True
        
        return min(base_score, 1.0)

class StageableHybridTransformer(nn.Module):
    """
    Hybrid Knowledge-Enhanced Transformer combining best of both approaches:
    - Neural efficiency from KGCapsule
    - Symbolic reasoning from PRSM  
    - Stageable complexity growth
    - Production-ready modularity
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        max_knowledge_nodes: int = 1000,
        initial_complexity_stage: int = 1,
        top_k_retrieval: int = 5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.top_k = top_k_retrieval
        
        # Hybrid knowledge management
        self.knowledge_manager = HybridKnowledgeManager(
            hidden_dim, max_knowledge_nodes
        )
        
        # Creative reasoning
        self.creative_layer = CreativeReasoningLayer()
        
        # Stage management
        self.current_stage = initial_complexity_stage
        self.stage_thresholds = {
            1: {"max_chunks": 3, "neural_top_k": 3, "symbolic_hops": 1},
            2: {"max_chunks": 5, "neural_top_k": 5, "symbolic_hops": 2}, 
            3: {"max_chunks": 8, "neural_top_k": 7, "symbolic_hops": 3},
            4: {"max_chunks": 12, "neural_top_k": 10, "symbolic_hops": 3}
        }
        
        # Neural processing layers (efficient like KGCapsule)
        self.knowledge_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.knowledge_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.knowledge_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Multi-modal fusion
        self.neural_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.symbolic_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Performance tracking
        self.success_rate_tracker = []
        self.stage_performance = {}
        
    def forward(
        self,
        sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_creative_reasoning: bool = True,
        update_knowledge: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with hybrid knowledge enhancement
        
        Args:
            sequence: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            use_creative_reasoning: Enable creative path generation
            update_knowledge: Enable knowledge updates during training
            
        Returns:
            enhanced_sequence: [batch_size, seq_len, hidden_dim]
            comprehensive_stats: Rich statistics dictionary
        """
        
        batch_size, seq_len, hidden_dim = sequence.shape
        device = sequence.device
        
        # Move knowledge to device
        self.knowledge_manager.neural_embeddings = self.knowledge_manager.neural_embeddings.to(device)
        self.knowledge_manager.neural_active = self.knowledge_manager.neural_active.to(device)
        
        # Get current stage configuration
        stage_config = self.stage_thresholds[self.current_stage]
        
        enhanced_tokens = []
        processing_stats = {
            'neural_retrievals': 0,
            'symbolic_inferences': 0,
            'creative_paths_used': 0,
            'knowledge_updates': 0,
            'avg_confidence': 0.0
        }
        
        # Process each token position
        total_confidence = 0.0
        for pos in range(seq_len):
            if attention_mask is not None and not attention_mask[:, pos].any():
                enhanced_tokens.append(sequence[:, pos, :])
                continue
            
            token_vectors = sequence[:, pos, :]
            
            # Enhanced processing with hybrid knowledge
            enhanced_token, token_stats = self._process_token_hybrid(
                token_vectors, stage_config, use_creative_reasoning
            )
            
            enhanced_tokens.append(enhanced_token)
            
            # Accumulate statistics
            processing_stats['neural_retrievals'] += token_stats.get('neural_matches', 0)
            processing_stats['symbolic_inferences'] += token_stats.get('symbolic_inferences', 0)
            processing_stats['creative_paths_used'] += token_stats.get('creative_paths', 0)
            total_confidence += token_stats.get('confidence', 0.0)
            
            # Update knowledge if enabled
            if update_knowledge and self.training:
                self._update_knowledge_from_token(token_vectors, token_stats)
                processing_stats['knowledge_updates'] += 1
        
        # Stack enhanced tokens
        enhanced_sequence = torch.stack(enhanced_tokens, dim=1)
        
        # Compute final statistics
        processing_stats['avg_confidence'] = total_confidence / seq_len
        
        # Check for stage advancement
        if len(self.success_rate_tracker) >= 10:
            recent_success = np.mean(self.success_rate_tracker[-10:])
            if recent_success > 0.85 and self.current_stage < max(self.stage_thresholds.keys()):
                self._advance_stage()
        
        # Comprehensive statistics
        comprehensive_stats = {
            **processing_stats,
            'current_stage': self.current_stage,
            'knowledge_nodes': self.knowledge_manager.current_neural_nodes,
            'symbolic_entities': len(self.knowledge_manager.symbolic_graph.nodes()),
            'symbolic_relations': len(self.knowledge_manager.symbolic_graph.edges()),
            'stage_config': stage_config
        }
        
        return enhanced_sequence, comprehensive_stats
    
    def _process_token_hybrid(
        self, 
        token_vectors: torch.Tensor, 
        stage_config: Dict,
        use_creative_reasoning: bool
    ) -> Tuple[torch.Tensor, Dict]:
        """Process tokens with hybrid neural + symbolic enhancement"""
        
        batch_size = token_vectors.shape[0]
        
        # Hybrid knowledge retrieval
        enhanced_results = []
        token_stats = {'neural_matches': 0, 'symbolic_inferences': 0, 'creative_paths': 0, 'confidence': 0.0}
        
        for i in range(batch_size):
            token_embedding = token_vectors[i]
            
            # Get hybrid knowledge context
            hybrid_context = self.knowledge_manager.retrieve_hybrid(
                token_embedding,
                top_k_neural=stage_config['neural_top_k'],
                max_symbolic_hops=stage_config['symbolic_hops']
            )
            
            token_stats['neural_matches'] += len(hybrid_context['neural_matches'])
            token_stats['symbolic_inferences'] += len(hybrid_context['symbolic_context'])
            
            # Neural attention mechanism
            enhanced_token = self._apply_neural_attention(
                token_embedding, hybrid_context['neural_matches']
            )
            
            # Symbolic reasoning enhancement
            if hybrid_context['symbolic_context'] and use_creative_reasoning:
                symbolic_enhancement = self._apply_symbolic_reasoning(
                    enhanced_token, hybrid_context['symbolic_context']
                )
                enhanced_token = self._fuse_enhancements(enhanced_token, symbolic_enhancement)
                token_stats['creative_paths'] += 1
            
            enhanced_results.append(enhanced_token)
            
            # Better confidence calculation - use similarity or generate base confidence
            if hybrid_context['neural_matches']:
                token_stats['confidence'] += hybrid_context['hybrid_score']
            else:
                # Generate base confidence for novel tokens
                token_norm = torch.norm(token_embedding).item()
                base_confidence = min(0.8, max(0.3, token_norm / 10.0))  # Normalize to 0.3-0.8 range
                token_stats['confidence'] += base_confidence
        
        # Stack results
        enhanced_tokens = torch.stack(enhanced_results, dim=0)
        token_stats['confidence'] /= batch_size
        
        return enhanced_tokens, token_stats
    
    def _apply_neural_attention(self, token_embedding: torch.Tensor, neural_matches: List[Dict]) -> torch.Tensor:
        """Apply neural attention over retrieved knowledge"""
        if not neural_matches:
            return token_embedding
        
        # Get knowledge embeddings
        knowledge_embeddings = []
        similarities = []
        
        for match in neural_matches:
            neural_idx = match['neural_idx']
            if neural_idx >= 0:
                knowledge_embeddings.append(
                    self.knowledge_manager.neural_embeddings[neural_idx]
                )
                similarities.append(match['similarity'])
        
        if not knowledge_embeddings:
            return token_embedding
        
        knowledge_stack = torch.stack(knowledge_embeddings, dim=0)  # [num_matches, hidden_dim]
        similarity_weights = torch.tensor(similarities, device=token_embedding.device)
        similarity_weights = F.softmax(similarity_weights, dim=0)
        
        # Attention mechanism
        query = self.knowledge_query(token_embedding.unsqueeze(0))  # [1, hidden_dim]
        keys = self.knowledge_key(knowledge_stack)  # [num_matches, hidden_dim]
        values = self.knowledge_value(knowledge_stack)  # [num_matches, hidden_dim]
        
        attention_scores = torch.matmul(query, keys.T) / np.sqrt(self.hidden_dim)  # [1, num_matches]
        attention_weights = F.softmax(attention_scores, dim=-1) * similarity_weights.unsqueeze(0)
        
        # Apply attention
        attended_knowledge = torch.matmul(attention_weights, values).squeeze(0)  # [hidden_dim]
        
        # Gate and combine
        combined = torch.cat([token_embedding, attended_knowledge], dim=0)
        gate = self.neural_gate(combined)
        
        return gate * attended_knowledge + (1 - gate) * token_embedding
    
    def _apply_symbolic_reasoning(self, token_embedding: torch.Tensor, symbolic_context: List[Dict]) -> torch.Tensor:
        """Apply symbolic reasoning enhancement"""
        if not symbolic_context:
            return torch.zeros_like(token_embedding)
        
        # Simple symbolic enhancement - could be made more sophisticated
        reasoning_strength = sum(rel['strength'] for rel in symbolic_context) / len(symbolic_context)
        reasoning_influence = torch.randn_like(token_embedding) * reasoning_strength * 0.1
        
        return self.symbolic_gate(token_embedding) * reasoning_influence
    
    def _fuse_enhancements(self, neural_enhanced: torch.Tensor, symbolic_enhanced: torch.Tensor) -> torch.Tensor:
        """Fuse neural and symbolic enhancements"""
        fused = neural_enhanced + symbolic_enhanced
        return self.output_projection(fused) + neural_enhanced  # Residual connection
    
    def _update_knowledge_from_token(self, token_vectors: torch.Tensor, token_stats: Dict):
        """Update knowledge from processed tokens"""
        # Progressive threshold - stricter as we learn more
        if self.knowledge_manager.current_neural_nodes < 5:
            confidence_threshold = 0.2  # Very permissive initially
        elif self.knowledge_manager.current_neural_nodes < 20:
            confidence_threshold = 0.4  # Moderate
        else:
            confidence_threshold = 0.6  # Stricter for quality
        
        current_confidence = token_stats.get('confidence', 0.0)
        
        if current_confidence >= confidence_threshold:
            for i in range(min(2, token_vectors.shape[0])):  # Limit to 2 tokens per batch
                # Create meaningful entity names
                timestamp = int(time.time() * 1000) % 100000
                entity_name = f"learned_concept_{timestamp}_{i}"
                
                # Determine concept type based on embedding characteristics
                embedding = token_vectors[i].detach()
                concept_type = self._classify_embedding_type(embedding)
                
                neural_idx = self.knowledge_manager.add_knowledge(
                    entity=entity_name,
                    embedding=embedding,
                    concept_type=concept_type,
                    confidence=current_confidence,
                    symbolic_data={'learning_round': len(self.success_rate_tracker)}
                )
                
                # Add relations to recent knowledge
                if neural_idx >= 0 and self.knowledge_manager.current_neural_nodes > 1:
                    recent_entities = list(self.knowledge_manager.entity_metadata.keys())[-3:]
                    for recent_entity in recent_entities:
                        if recent_entity != entity_name:
                            self.knowledge_manager.add_relation(
                                entity_name, recent_entity, "co_occurred", 0.7
                            )
        elif self.knowledge_manager.current_neural_nodes < 3:
            # Force learning for first few examples
            entity_name = f"forced_learning_{int(time.time() * 1000) % 10000}"
            self.knowledge_manager.add_knowledge(
                entity=entity_name,
                embedding=token_vectors[0].detach(),
                concept_type="forced",
                confidence=0.8,
                symbolic_data={'forced': True}
            )
    
    def _classify_embedding_type(self, embedding: torch.Tensor) -> str:
        """Classify embedding into concept types based on characteristics"""
        # Simple heuristic classification
        mean_val = embedding.mean().item()
        std_val = embedding.std().item()
        
        if abs(mean_val) > 0.1:
            return "distinctive"
        elif std_val > 0.5:
            return "complex"
        elif std_val < 0.2:
            return "simple"
        else:
            return "general"
    
    def _advance_stage(self):
        """Advance to next complexity stage"""
        self.current_stage += 1
        print(f"🚀 Advanced to Stage {self.current_stage}")
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'stage': self.current_stage,
            'neural_knowledge': {
                'nodes': self.knowledge_manager.current_neural_nodes,
                'max_capacity': self.knowledge_manager.max_neural_nodes,
                'utilization': self.knowledge_manager.current_neural_nodes / self.knowledge_manager.max_neural_nodes
            },
            'symbolic_knowledge': {
                'entities': len(self.knowledge_manager.symbolic_graph.nodes()),
                'relations': len(self.knowledge_manager.symbolic_graph.edges()),
                'concepts': len(self.knowledge_manager.concept_hierarchy)
            },
            'creative_reasoning': {
                'patterns_stored': len(self.creative_layer.creative_patterns),
                'reasoning_cache_size': len(self.creative_layer.reasoning_cache)
            },
            'recent_performance': np.mean(self.success_rate_tracker[-10:]) if self.success_rate_tracker else 0.0
        }


# Enhanced demonstration function with multiple rounds
def demonstrate_hybrid_system():
    """Demonstrate the hybrid system capabilities with progressive learning"""
    print("🚀 Hybrid Knowledge-Enhanced Transformer (HKET) Demo")
    print("=" * 60)
    
    # Initialize system
    hket = StageableHybridTransformer(
        hidden_dim=768,
        max_knowledge_nodes=500,
        initial_complexity_stage=1
    )
    
    # Test multiple rounds to show learning progression
    batch_size, seq_len, hidden_dim = 2, 8, 768
    
    print("📚 Progressive Learning Demonstration:")
    print("-" * 40)
    
    for round_num in range(5):
        print(f"\n🔄 Round {round_num + 1}:")
        
        # Generate different test data each round
        test_sequence = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Process with hybrid enhancement
        enhanced_sequence, stats = hket(
            test_sequence, 
            attention_mask,
            use_creative_reasoning=True,
            update_knowledge=True
        )
        
        # Show progression
        system_stats = hket.get_comprehensive_stats()
        print(f"   Knowledge Nodes: {stats['knowledge_nodes']}")
        print(f"   Neural Retrievals: {stats['neural_retrievals']}")
        print(f"   Symbolic Relations: {stats['symbolic_relations']}")
        print(f"   Confidence: {stats.get('avg_confidence', 0.0):.3f}")
        print(f"   Updates Made: {stats.get('knowledge_updates', 0)}")
        print(f"   Stage: {stats['current_stage']}")
        
        # Add some explicit relations for demonstration
        if round_num > 0 and stats['knowledge_nodes'] > 1:
            # Manually add some relations to show symbolic reasoning
            entities = list(hket.knowledge_manager.entity_metadata.keys())
            if len(entities) >= 2:
                hket.knowledge_manager.add_relation(
                    entities[0], entities[-1], "learned_together", 0.8
                )
                print(f"   ➕ Added relation: {entities[0]} → {entities[-1]}")
    
    print(f"\n🎉 FINAL SYSTEM STATE:")
    print("=" * 40)
    
    final_stats = hket.get_comprehensive_stats()
    for category, data in final_stats.items():
        if isinstance(data, dict):
            print(f"📊 {category.upper()}:")
            for key, value in data.items():
                print(f"   {key}: {value}")
        else:
            print(f"📈 {category}: {data}")
    
    # Demonstrate retrieval capabilities
    print(f"\n🔍 Testing Knowledge Retrieval:")
    test_query = torch.randn(768)
    if hket.knowledge_manager.current_neural_nodes > 0:
        retrieval_result = hket.knowledge_manager.retrieve_hybrid(test_query, top_k_neural=3)
        print(f"   Found {len(retrieval_result['neural_matches'])} neural matches")
        print(f"   Found {len(retrieval_result['symbolic_context'])} symbolic connections")
        print(f"   Hybrid score: {retrieval_result['hybrid_score']:.3f}")
        
        if retrieval_result['neural_matches']:
            top_match = retrieval_result['neural_matches'][0]
            print(f"   Top match: {top_match['entity']} (similarity: {top_match['similarity']:.3f})")
    
    return hket


if __name__ == "__main__":
    system = demonstrate_hybrid_system()
    print("\n✅ Hybrid system demonstration completed!")
    print("💡 Combines neural efficiency with symbolic reasoning!")
