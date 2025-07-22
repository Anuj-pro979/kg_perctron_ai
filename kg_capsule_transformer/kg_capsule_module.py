import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any

class KGCapsuleTransformer(nn.Module):
    """
    Knowledge Graph Capsule for Transformer Sandwich Architecture
    
    Designed to fit between frozen encoder and decoder:
    Encoder → KGCapsule → Decoder
    
    Features:
    - Sequence-aware processing
    - Controlled knowledge updates
    - Memory-efficient sparse storage
    - Batch-optimized operations
    """
    
    def __init__(
        self, 
        hidden_dim: int = 768,
        max_knowledge_nodes: int = 1000,
        top_k_retrieval: int = 5,
        confidence_threshold: float = 0.7,
        knowledge_update_rate: float = 0.1
    ):
        super().__init__()
        
        # Core parameters
        self.hidden_dim = hidden_dim
        self.max_nodes = max_knowledge_nodes
        self.top_k = top_k_retrieval
        self.confidence_threshold = confidence_threshold
        self.update_rate = knowledge_update_rate
        self.current_nodes = 0
        
        # Knowledge storage (sparse and efficient)
        self.knowledge_embeddings = nn.Parameter(
            torch.randn(max_knowledge_nodes, hidden_dim) * 0.02
        )
        self.node_active = nn.Parameter(
            torch.zeros(max_knowledge_nodes, dtype=torch.bool), 
            requires_grad=False
        )
        self.node_usage_count = torch.zeros(max_knowledge_nodes)
        
        # Knowledge graph connections (adjacency matrix)
        self.adjacency_weights = nn.Parameter(
            torch.zeros(max_knowledge_nodes, max_knowledge_nodes),
            requires_grad=False
        )
        
        # Neural processing layers
        self.knowledge_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.knowledge_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.knowledge_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Knowledge integration
        self.knowledge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Update buffer for async knowledge management
        self.update_buffer = []
        self.max_buffer_size = 500
        
        # Initialize with basic knowledge structure
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Initialize with a basic knowledge node"""
        with torch.no_grad():
            self.node_active[0] = True
            self.current_nodes = 1
    
    def forward(
        self, 
        sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_knowledge: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with sequence processing
        
        Args:
            sequence: [batch_size, seq_len, hidden_dim] from encoder
            attention_mask: [batch_size, seq_len] padding mask
            update_knowledge: Whether to update knowledge during training
            
        Returns:
            enhanced_sequence: [batch_size, seq_len, hidden_dim] for decoder
            stats: Dictionary with processing statistics
        """
        batch_size, seq_len, hidden_dim = sequence.shape
        device = sequence.device
        
        # Ensure tensors are on correct device
        if self.knowledge_embeddings.device != device:
            self = self.to(device)
            self.node_usage_count = self.node_usage_count.to(device)
        
        # Get active knowledge embeddings
        active_mask = self.node_active
        active_knowledge = self.knowledge_embeddings[active_mask]  # [active_nodes, hidden_dim]
        
        if active_knowledge.shape[0] == 0:
            # No knowledge yet - return original sequence
            return sequence, {"active_nodes": 0, "knowledge_retrieved": False}
        
        # Process sequence with knowledge retrieval
        enhanced_tokens = []
        total_confidence = 0.0
        knowledge_used = 0
        
        for pos in range(seq_len):
            # Skip padding tokens
            if attention_mask is not None and not attention_mask[:, pos].any():
                enhanced_tokens.append(sequence[:, pos, :])
                continue
            
            token_vectors = sequence[:, pos, :]  # [batch_size, hidden_dim]
            enhanced_token, confidence = self._process_token_with_knowledge(
                token_vectors, active_knowledge
            )
            
            enhanced_tokens.append(enhanced_token)
            total_confidence += confidence.mean().item()
            knowledge_used += 1
            
            # Queue knowledge updates for high-confidence novel patterns
            if update_knowledge and self.training and confidence.mean() > self.confidence_threshold:
                self._queue_knowledge_update(token_vectors.detach(), confidence.detach())
        
        # Stack processed tokens
        enhanced_sequence = torch.stack(enhanced_tokens, dim=1)  # [batch, seq_len, hidden_dim]
        
        # Process queued knowledge updates
        updates_made = 0
        if update_knowledge and self.training:
            updates_made = self._process_knowledge_updates()
        
        # Compile statistics
        stats = {
            "active_nodes": active_knowledge.shape[0],
            "knowledge_retrieved": knowledge_used > 0,
            "avg_confidence": total_confidence / max(knowledge_used, 1),
            "updates_made": updates_made,
            "buffer_size": len(self.update_buffer)
        }
        
        return enhanced_sequence, stats
    
    def _process_token_with_knowledge(
        self, 
        token_vectors: torch.Tensor, 
        active_knowledge: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process individual token position with knowledge retrieval
        
        Args:
            token_vectors: [batch_size, hidden_dim]
            active_knowledge: [active_nodes, hidden_dim]
            
        Returns:
            enhanced_tokens: [batch_size, hidden_dim]
            confidence: [batch_size]
        """
        batch_size = token_vectors.shape[0]
        
        # Knowledge retrieval via attention mechanism
        queries = self.knowledge_query(token_vectors)  # [batch, hidden_dim]
        keys = self.knowledge_key(active_knowledge)    # [active_nodes, hidden_dim]
        values = self.knowledge_value(active_knowledge) # [active_nodes, hidden_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.T) / np.sqrt(self.hidden_dim)  # [batch, active_nodes]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Top-k knowledge selection
        k = min(self.top_k, active_knowledge.shape[0])
        topk_weights, topk_indices = torch.topk(attention_weights, k, dim=-1)  # [batch, k]
        
        # Retrieve and aggregate top-k knowledge
        selected_values = values[topk_indices]  # [batch, k, hidden_dim]
        weighted_knowledge = selected_values * topk_weights.unsqueeze(-1)  # [batch, k, hidden_dim]
        aggregated_knowledge = weighted_knowledge.sum(dim=1)  # [batch, hidden_dim]
        
        # Update node usage statistics
        with torch.no_grad():
            for batch_idx in range(batch_size):
                for i in range(k):
                    node_idx = topk_indices[batch_idx, i].item()
                    self.node_usage_count[node_idx] += topk_weights[batch_idx, i].item()
        
        # Knowledge integration with gating mechanism
        combined = torch.cat([token_vectors, aggregated_knowledge], dim=-1)  # [batch, hidden_dim*2]
        gate = self.knowledge_gate(combined)  # [batch, hidden_dim]
        
        # Gated combination
        enhanced_tokens = gate * aggregated_knowledge + (1 - gate) * token_vectors
        
        # Apply output projection with residual connection
        enhanced_tokens = self.output_projection(enhanced_tokens) + token_vectors
        
        # Compute confidence based on attention concentration
        confidence = topk_weights.max(dim=-1)[0]  # [batch]
        
        return enhanced_tokens, confidence
    
    def _queue_knowledge_update(self, token_vectors: torch.Tensor, confidence: torch.Tensor):
        """Queue token vectors for potential knowledge node creation"""
        for i in range(token_vectors.shape[0]):
            if len(self.update_buffer) < self.max_buffer_size:
                self.update_buffer.append({
                    'embedding': token_vectors[i].clone(),
                    'confidence': confidence[i].item()
                })
    
    def _process_knowledge_updates(self) -> int:
        """Process queued knowledge updates"""
        if not self.update_buffer or self.current_nodes >= self.max_nodes:
            return 0
        
        updates_made = 0
        max_updates_per_batch = min(5, len(self.update_buffer))
        
        for _ in range(max_updates_per_batch):
            if not self.update_buffer:
                break
                
            update_data = self.update_buffer.pop(0)
            
            if self._should_create_knowledge_node(update_data['embedding']):
                self._create_knowledge_node(update_data['embedding'])
                updates_made += 1
        
        return updates_made
    
    def _should_create_knowledge_node(self, embedding: torch.Tensor) -> bool:
        """Determine if a new knowledge node should be created"""
        if self.current_nodes >= self.max_nodes:
            return False
        
        active_knowledge = self.knowledge_embeddings[self.node_active]
        if active_knowledge.shape[0] == 0:
            return True
        
        # Check similarity to existing knowledge
        similarities = F.cosine_similarity(embedding.unsqueeze(0), active_knowledge, dim=-1)
        max_similarity = similarities.max()
        
        # Create node if sufficiently novel
        return max_similarity < 0.85
    
    def _create_knowledge_node(self, embedding: torch.Tensor) -> int:
        """Create a new knowledge node"""
        if self.current_nodes >= self.max_nodes:
            return -1
        
        node_idx = self.current_nodes
        
        with torch.no_grad():
            self.knowledge_embeddings[node_idx] = embedding
            self.node_active[node_idx] = True
            
            # Connect to similar existing nodes
            if self.current_nodes > 0:
                active_knowledge = self.knowledge_embeddings[self.node_active]
                similarities = F.cosine_similarity(embedding.unsqueeze(0), active_knowledge, dim=-1)
                
                # Connect to most similar nodes above threshold
                for i, sim in enumerate(similarities):
                    if sim > 0.6:
                        self.adjacency_weights[node_idx, i] = sim.item()
                        self.adjacency_weights[i, node_idx] = sim.item() * 0.8
        
        self.current_nodes += 1
        return node_idx
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get comprehensive knowledge statistics"""
        active_nodes = self.node_active.sum().item()
        total_connections = (self.adjacency_weights > 0).sum().item()
        avg_usage = self.node_usage_count[self.node_active].mean().item() if active_nodes > 0 else 0
        
        return {
            'total_nodes': active_nodes,
            'max_nodes': self.max_nodes,
            'utilization_ratio': active_nodes / self.max_nodes,
            'total_connections': total_connections,
            'avg_node_usage': avg_usage,
            'pending_updates': len(self.update_buffer),
            'memory_efficient': True
        }
    
    def prune_unused_knowledge(self, usage_threshold: float = 0.01):
        """Remove rarely used knowledge nodes"""
        if self.current_nodes <= 1:
            return 0
        
        with torch.no_grad():
            # Find nodes with low usage
            avg_usage = self.node_usage_count[self.node_active].mean()
            low_usage_mask = self.node_usage_count < (avg_usage * usage_threshold)
            nodes_to_remove = self.node_active & low_usage_mask
            
            # Keep at least one node
            if nodes_to_remove.sum() >= self.current_nodes:
                return 0
            
            # Remove low-usage nodes
            self.node_active[nodes_to_remove] = False
            self.node_usage_count[nodes_to_remove] = 0
            
            # Clean up adjacency matrix
            self.adjacency_weights[nodes_to_remove, :] = 0
            self.adjacency_weights[:, nodes_to_remove] = 0
            
            removed_count = nodes_to_remove.sum().item()
            self.current_nodes -= removed_count
            
            return removed_count
    
    def clear_update_buffer(self):
        """Clear the knowledge update buffer"""
        self.update_buffer.clear()


# Example usage and testing
def test_kg_capsule():
    """Test function for the KGCapsule module"""
    print("Testing KGCapsule Transformer Module...")
    
    # Initialize
    kg_capsule = KGCapsuleTransformer(
        hidden_dim=768,
        max_knowledge_nodes=100,
        top_k_retrieval=3
    )
    
    # Test input (typical transformer sequence)
    batch_size, seq_len, hidden_dim = 4, 16, 768
    test_sequence = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    print(f"Input shape: {test_sequence.shape}")
    
    # Forward pass
    enhanced_sequence, stats = kg_capsule(test_sequence, attention_mask)
    
    print(f"Output shape: {enhanced_sequence.shape}")
    print(f"Processing stats: {stats}")
    print(f"Knowledge stats: {kg_capsule.get_knowledge_stats()}")
    
    # Test knowledge accumulation over multiple batches
    print("\nTesting knowledge accumulation...")
    kg_capsule.train()
    
    for i in range(5):
        batch = torch.randn(2, 8, 768)
        _, stats = kg_capsule(batch, update_knowledge=True)
        print(f"Batch {i+1}: {stats['active_nodes']} nodes, {stats['updates_made']} updates")
    
    print(f"Final knowledge stats: {kg_capsule.get_knowledge_stats()}")
    print("✅ KGCapsule test completed!")


if __name__ == "__main__":
    test_kg_capsule()