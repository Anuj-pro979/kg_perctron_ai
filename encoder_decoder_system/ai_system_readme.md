# Modular AI System with Knowledge Capsules

A complete modular AI architecture that uses knowledge capsules for intelligent query processing and response generation.

## 🏗️ Architecture

```
User Query → ModularEncoder → KnowledgeCapsulesAlgorithm → TransformerDecoder → Final Response
     ↓              ↓                    ↓                       ↓
Text Input    Query Vector    Active Capsules + Vectors    Smart Response
```

## 🧩 Components

### 1. **ModularEncoder** (`modular_encoder.py`)
- Converts text to vector embeddings using SentenceTransformer
- Handles similarity calculations between texts and vectors
- Provides consistent encoding across the system

### 2. **KnowledgeCapsulesAlgorithm** (`updated_capsules_algorithm.py`)
- 5-step processing algorithm:
  1. Calculate similarities between query and capsules
  2. Apply activation function (threshold-based)
  3. Calculate inter-capsule signals (connections)
  4. Compute final activations (similarity + signals)
  5. Generate response from active capsules
- Learns from feedback by strengthening/weakening connections

### 3. **EmbeddedNeo4jConnection** (`updated_neo4j_setup.py`)
- In-memory graph database for knowledge capsules
- Auto-creates connections between similar capsules
- Handles persistence and capsule management

### 4. **TransformerDecoder** (`transformer_decoder.py`)
- Uses GPT-2 for advanced response generation
- Falls back to MinimalDecoder if transformer fails
- Processes query + capsule vectors for contextual responses

### 5. **ModularAISystem** (`ai_system_integration.py`)
- Main integration point connecting all components
- Handles the complete processing pipeline
- Manages system lifecycle and feedback learning

## 🚀 Quick Start

### Installation
```bash
pip install sentence-transformers transformers torch numpy scikit-learn
```

### Basic Usage
```python
from ai_system_integration import ModularAISystem

# Initialize system
ai = ModularAISystem()

# Add knowledge
ai.add_knowledge("health_001", "Exercise improves cardiovascular health", "health")
ai.add_knowledge("tech_001", "Python is great for AI development", "technology")

# Process queries
response, metadata = ai.process_query("How can I improve my health?")
print(f"Response: {response}")

# Learn from feedback
ai.learn_from_feedback("good response")

# Close system
ai.close()
```

### Run Demo
```bash
python ai_system_integration.py
```

## 🔄 How It Works

1. **Query Processing**: User query is converted to vector embedding
2. **Capsule Activation**: System finds relevant knowledge capsules based on similarity
3. **Signal Propagation**: Connected capsules boost each other's activation
4. **Response Generation**: Decoder uses query vector + active capsule vectors to generate response
5. **Learning**: System adjusts capsule connections based on feedback

## 📊 Key Features

- **Modular Design**: Each component is independent and replaceable
- **Vector-Based**: Uses embeddings for semantic understanding
- **Graph Structure**: Capsules form a connected knowledge network
- **Learning Capability**: Improves through feedback mechanisms
- **Fallback Systems**: Graceful degradation if components fail
- **Persistent Storage**: Saves knowledge between sessions

## 🎯 Example Output

```
🔄 Processing: 'How can I improve my health?'
📊 Query encoded to 384 dimensions
📊 Calculating similarities...
  health_001: 0.847
  tech_001: 0.234
⚡ Applying activation...
  ✅ health_001: ACTIVE (0.847)
  💤 tech_001: inactive (0.234)
🔊 Calculating signals...
  📡 health_001 → health_002: 0.523
💡 Final activations...
  🧮 health_001: 0.847 + 0.000 = 0.847
  🧮 health_002: 0.234 + 0.523 = 0.757
🔥 Active capsules:
  ✅ health_001: 0.847
  ✅ health_002: 0.757
🤖 Final response generated through decoder using vector embeddings

💬 Response: To address your question: Exercise improves cardiovascular health. Mental health is improved through physical exercise and proper nutrition. This approach should help you achieve your goal.

[Confidence: High | Sources: 2 capsules | Categories: health]
```

## 🛠️ System Status

Check system health:
```python
ai.get_system_status()
# Output: {'total_capsules': 4, 'active_capsules': 2, 'encoder_dimensions': 384, ...}

ai.show_active_knowledge()
# Shows currently active capsules with activation scores
```

## 📁 File Structure

```
├── ai_system_integration.py      # Main system integration
├── modular_encoder.py            # Text encoding module
├── updated_capsules_algorithm.py # Knowledge processing
├── updated_neo4j_setup.py        # Database management
├── transformer_decoder.py        # Response generation
└── README.md                     # This file
```

## 🎓 Use Cases

- **Question Answering**: Intelligent responses based on knowledge base
- **Educational Systems**: Adaptive learning with feedback
- **Knowledge Management**: Semantic search and retrieval
- **Chatbots**: Context-aware conversational AI
- **Research Assistant**: Domain-specific knowledge processing

## 💡 Advanced Features

- **Bi-directional Learning**: Connections strengthen/weaken based on feedback
- **Semantic Similarity**: Uses state-of-the-art embeddings
- **Modular Architecture**: Easy to extend and modify
- **Vector Operations**: Rich mathematical operations on embeddings
- **Graph Intelligence**: Knowledge capsules form intelligent networks

---

**Ready to build intelligent AI systems with knowledge capsules!** 🚀