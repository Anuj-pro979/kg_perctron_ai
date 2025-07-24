# KGCapsule Transformer 🧠

A **Knowledge Graph Capsule** module designed to act as an adaptive memory layer between frozen encoder and decoder in transformer architectures. This module enables dynamic knowledge acquisition and contextual retrieval without retraining the entire model.

## 🎯 Overview

KGCapsule addresses the limitation of frozen transformer models by providing a learnable knowledge layer that can:
- **Dynamically learn** and store important patterns from data
- **Retrieve relevant knowledge** contextually for each token
- **Update its knowledge base** with novel patterns over time
- **Maintain memory efficiency** through sparse storage and pruning

## 🏗️ Architecture

```
Input Sequence → KGCapsule → Enhanced Sequence
     ↓              ↓              ↓
[Batch, Seq, Dim] → Knowledge → [Batch, Seq, Dim]
                   Retrieval &
                   Integration
```

### Core Components

1. **Knowledge Storage**: Sparse embedding matrix for storing learned patterns
2. **Retrieval System**: Attention-based mechanism for finding relevant knowledge
3. **Integration Layer**: Gated fusion of original tokens with retrieved knowledge
4. **Update Manager**: Dynamic node creation and pruning system

## 🔄 Algorithm Flow

### Phase 1: Initialization
```
┌─────────────────┐
│ Initialize      │
│ - Knowledge     │
│   embeddings    │
│ - Active nodes  │
│ - Adjacency     │
│   matrix        │
└─────────────────┘
```

### Phase 2: Forward Pass (For each token)

```
Input Token
     ↓
┌─────────────────┐
│ 1. Query        │
│    Generation   │
│    token → q    │
└─────────────────┘
     ↓
┌─────────────────┐
│ 2. Knowledge    │
│    Retrieval    │
│    q × K → attn │
└─────────────────┘
     ↓
┌─────────────────┐
│ 3. Top-K        │
│    Selection    │
│    topk(attn)   │
└─────────────────┘
     ↓
┌─────────────────┐
│ 4. Knowledge    │
│    Aggregation  │
│    Σ(weights×V) │
└─────────────────┘
     ↓
┌─────────────────┐
│ 5. Gated        │
│    Integration  │
│    gate×know +  │
│    (1-gate)×tok │
└─────────────────┘
     ↓
Enhanced Token
```

### Phase 3: Knowledge Updates (Training Mode)

```
High Confidence Novel Pattern?
     ↓ Yes
┌─────────────────┐
│ Queue for       │
│ Update Buffer   │
└─────────────────┘
     ↓
┌─────────────────┐
│ Batch Process   │
│ Updates         │
│ - Check novelty │
│ - Create nodes  │
│ - Build links   │
└─────────────────┘
```

## 🚀 Key Features

### Dynamic Knowledge Management
- **Adaptive Storage**: Creates new knowledge nodes for novel patterns
- **Similarity-based Connections**: Links related knowledge nodes
- **Usage Tracking**: Monitors node utilization for optimization
- **Automatic Pruning**: Removes rarely used nodes to prevent bloat

### Memory Efficiency
- **Sparse Storage**: Only active nodes consume memory
- **Batch Processing**: Efficient update handling
- **Top-K Retrieval**: Limits computational overhead

### Seamless Integration
- **Transformer Compatible**: Maintains sequence dimensions
- **Differentiable**: Full gradient flow for end-to-end training
- **Configurable**: Adjustable parameters for different use cases

## 📊 Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Knowledge Retrieval | O(seq_len × active_nodes) | O(active_nodes × hidden_dim) |
| Top-K Selection | O(seq_len × k × log(active_nodes)) | O(k) |
| Knowledge Updates | O(buffer_size × active_nodes) | O(buffer_size) |
| Node Pruning | O(active_nodes) | O(1) |

## 🛠️ Usage

### Basic Implementation

```python
from kg_capsule import KGCapsuleTransformer

# Initialize the module
kg_capsule = KGCapsuleTransformer(
    hidden_dim=768,
    max_knowledge_nodes=1000,
    top_k_retrieval=5,
    confidence_threshold=0.7,
    knowledge_update_rate=0.1
)

# Process sequences
batch_size, seq_len, hidden_dim = 4, 16, 768
input_sequence = torch.randn(batch_size, seq_len, hidden_dim)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

# Forward pass
enhanced_sequence, stats = kg_capsule(
    input_sequence, 
    attention_mask, 
    update_knowledge=True
)

print(f"Enhanced shape: {enhanced_sequence.shape}")
print(f"Processing stats: {stats}")
```

### Integration with Transformer

```python
class TransformerWithKGCapsule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = FrozenEncoder(config)  # Pre-trained
        self.kg_capsule = KGCapsuleTransformer(config.hidden_dim)
        self.decoder = FrozenDecoder(config)  # Pre-trained
    
    def forward(self, input_ids, attention_mask):
        # Encoder (frozen)
        encoded = self.encoder(input_ids, attention_mask)
        
        # KG Capsule (learnable)
        enhanced, stats = self.kg_capsule(encoded, attention_mask)
        
        # Decoder (frozen)
        output = self.decoder(enhanced, attention_mask)
        
        return output, stats
```

## 📈 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 768 | Dimension of hidden representations |
| `max_knowledge_nodes` | 1000 | Maximum number of knowledge nodes |
| `top_k_retrieval` | 5 | Number of top knowledge nodes to retrieve |
| `confidence_threshold` | 0.7 | Threshold for knowledge updates |
| `knowledge_update_rate` | 0.1 | Rate of knowledge incorporation |

## 🔍 Monitoring and Statistics

The module provides comprehensive statistics for monitoring:

```python
# Processing statistics
stats = {
    'active_nodes': int,        # Number of active knowledge nodes
    'knowledge_retrieved': bool, # Whether knowledge was used
    'avg_confidence': float,    # Average retrieval confidence
    'updates_made': int,        # Number of knowledge updates
    'buffer_size': int          # Size of update buffer
}

# Knowledge statistics
knowledge_stats = kg_capsule.get_knowledge_stats()
# Returns: total_nodes, utilization_ratio, total_connections, etc.
```

## 🧪 Testing

Run the included test function to verify functionality:

```bash
python kg_capsule.py
```

Expected output:
```
Testing KGCapsule Transformer Module...
Input shape: torch.Size([4, 16, 768])
Output shape: torch.Size([4, 16, 768])
Processing stats: {'active_nodes': 1, 'knowledge_retrieved': True, ...}
```

## 🎯 Use Cases

### 1. Domain Adaptation
- Add domain-specific knowledge to pre-trained models
- Maintain base model performance while specializing

### 2. Few-Shot Learning
- Learn from limited examples through knowledge accumulation
- Transfer patterns across similar tasks

### 3. Continual Learning
- Accumulate knowledge from multiple tasks
- Prevent catastrophic forgetting through selective storage

### 4. Knowledge-Augmented Generation
- Enhance text generation with learned patterns
- Improve consistency and factual accuracy

## ⚙️ PyTorch Components Utilized

### Core Neural Layers
- `nn.Linear` - Query/Key/Value projections and gating
- `nn.Parameter` - Learnable knowledge embeddings
- `nn.LayerNorm` - Output normalization
- `nn.Sigmoid` - Gating mechanism

### Tensor Operations
- `F.softmax` - Attention weight computation
- `F.cosine_similarity` - Knowledge similarity measurement
- `torch.topk` - Efficient top-k retrieval
- `torch.stack` - Sequence reconstruction

### Memory Management
- `torch.no_grad()` - Efficient non-gradient operations
- Automatic differentiation for gradient computation
- GPU/CPU tensor management

## 🚀 Advantages Over Manual Implementation

| Aspect | Manual Implementation | With PyTorch |
|--------|----------------------|--------------|
| Gradient Computation | Manual backprop coding | ✅ Automatic differentiation |
| Tensor Operations | Custom CUDA kernels | ✅ Optimized operations |
| Memory Management | Manual GPU handling | ✅ Automatic memory management |
| Batch Processing | Custom batching logic | ✅ Built-in batch support |
| Optimization | Custom optimizers | ✅ Ready-to-use optimizers |

## 🔧 Advanced Features

### Knowledge Pruning
```python
# Remove underutilized knowledge nodes
removed_count = kg_capsule.prune_unused_knowledge(usage_threshold=0.01)
print(f"Pruned {removed_count} nodes")
```

### Buffer Management
```python
# Clear update buffer if needed
kg_capsule.clear_update_buffer()
```

### Knowledge Visualization
```python
# Get detailed knowledge statistics
stats = kg_capsule.get_knowledge_stats()
print(f"Knowledge utilization: {stats['utilization_ratio']:.2%}")
print(f"Average node usage: {stats['avg_node_usage']:.3f}")
```

## 📝 Technical Notes

### Memory Considerations
- Sparse storage reduces memory footprint
- Active node tracking prevents unnecessary computations
- Buffer size limits memory usage during training

### Performance Optimization
- Top-k retrieval limits computational complexity
- Batch processing of updates improves efficiency
- Usage-based pruning maintains model size

### Gradient Flow
- Full differentiability enables end-to-end training
- Gating mechanism allows for selective gradient flow
- Knowledge updates are performed in no-grad context for efficiency

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional similarity metrics for knowledge retrieval
- More sophisticated pruning strategies
- Integration with different transformer architectures
- Performance optimizations for large-scale deployment

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This module is designed as a research prototype. For production use, consider additional optimizations and thorough testing with your specific use case.
