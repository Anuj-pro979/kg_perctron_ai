# Knowledge Capsules - Intelligent Q&A System

🧠 An advanced knowledge management and question-answering system that uses interconnected "knowledge capsules" to provide intelligent responses through neural-inspired processing.

## 🌟 Overview

The Knowledge Capsules system is inspired by neural networks and creates a dynamic, interconnected knowledge base where information is stored in discrete "capsules" that can activate and influence each other based on semantic similarity and learned connections.

## 🏗️ Architecture

### Core Components

1. **Sentence Encoder** (`SentenceTransformer`)
   - Converts text into high-dimensional vectors (embeddings)
   - Uses `all-MiniLM-L6-v2` model for semantic understanding
   - Enables similarity calculations between queries and knowledge

2. **Knowledge Capsules** (Graph Database)
   - Individual knowledge units containing text + vector representation
   - Auto-connected based on semantic similarity
   - Can be activated and deactivated dynamically

3. **Processing Algorithm** (5-Step Neural-Inspired Process)
   - Calculates similarities, applies activation functions
   - Implements inter-capsule communication
   - Generates intelligent responses

4. **Learning Mechanism**
   - Adapts connection weights based on user feedback
   - Strengthens successful knowledge pathways
   - Weakens ineffective connections

## 🔄 How Encoder and Capsules Work Together

### 1. **Vector Generation & Storage**
```
User Input Text → Sentence Encoder → Vector [0.2, -0.1, 0.8, ...]
                                        ↓
Knowledge Capsule: {
    text: "Exercise strengthens muscles...",
    vector: [0.2, -0.1, 0.8, ...],
    category: "health"
}
```

### 2. **Similarity Calculation**
```
Query Vector    [0.3, 0.1, 0.7, ...]
Capsule Vector  [0.2, -0.1, 0.8, ...]
                        ↓
Cosine Similarity = 0.85 (High similarity!)
```

### 3. **Dynamic Activation**
- Capsules with similarity > threshold (0.4) become ACTIVE
- Active capsules can influence connected capsules
- Creates cascading activation patterns

### 4. **Inter-Capsule Communication**
```
exercise (active: 0.85) → nutrition (connected: weight 0.6)
Signal Boost = 0.85 × 0.6 = 0.51
nutrition final activation = original + boost
```

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd knowledge-capsules

# Install dependencies
pip install -r requirements.txt

# Required packages:
pip install sentence-transformers
pip install scikit-learn
pip install numpy
pip install neo4j  # Optional for future graph database integration
```

## 🚀 Quick Start

### Basic Usage

```python
from capsules_app import KnowledgeCapsulesApp

# Initialize the system
app = KnowledgeCapsulesApp()

# Ask questions
response = app.ask_question("How can I stay healthy?")
print(response)

# Provide feedback to improve learning
app.give_feedback("good")  # or "bad"

# Add custom knowledge
app.add_custom_capsule(
    "meditation", 
    "Meditation reduces stress and improves focus through mindfulness practice",
    "wellness"
)

# Check system status
app.show_system_status()
```

### Interactive Session

```python
# Run interactive Q&A session
app.interactive_session()
```

### Command Line Usage

```bash
python capsules_app.py
```

## 🧬 Algorithm Deep Dive

### 5-Step Processing Algorithm

#### Step 1: Similarity Calculation
- Convert user query to vector using sentence encoder
- Calculate cosine similarity with all knowledge capsules
- Results in similarity scores (0.0 to 1.0)

#### Step 2: Activation Function
- Apply threshold filter (default: 0.4)
- Capsules above threshold become "active"
- Inactive capsules set to 0.0 activation

#### Step 3: Inter-Capsule Signaling
- Active capsules send signals to connected capsules
- Signal strength = activation × connection_weight
- Simulates neural network propagation

#### Step 4: Final Activation Calculation
- Combine original similarity + signal boosts
- `final_activation = similarity + Σ(incoming_signals)`
- Creates emergent activation patterns

#### Step 5: Response Generation
- Rank capsules by final activation
- Select top 3 most active capsules
- Combine their knowledge into coherent response

## 📊 Example Walkthrough

**Query**: "How can I stay healthy?"

```
1. Similarity Calculation:
   - exercise: 0.85
   - nutrition: 0.72
   - sleep: 0.68
   - programming: 0.15

2. Activation (threshold=0.4):
   - ✅ exercise: 0.85 (ACTIVE)
   - ✅ nutrition: 0.72 (ACTIVE) 
   - ✅ sleep: 0.68 (ACTIVE)
   - ❌ programming: 0.0 (inactive)

3. Signal Calculation:
   - exercise → nutrition: 0.85 × 0.6 = 0.51
   - nutrition → sleep: 0.72 × 0.4 = 0.29

4. Final Activations:
   - exercise: 0.85 + 0.0 = 0.85
   - nutrition: 0.72 + 0.51 = 1.23
   - sleep: 0.68 + 0.29 = 0.97

5. Response Generation:
   Top capsules: nutrition (1.23), sleep (0.97), exercise (0.85)
   Generated response combining all three knowledge areas.
```

## 🎯 Key Features

### Adaptive Learning
- **Positive Feedback**: Strengthens connections between co-activated capsules
- **Negative Feedback**: Weakens problematic connections
- **Continuous Improvement**: System gets better with usage

### Semantic Understanding
- Uses state-of-the-art sentence transformers
- Captures meaning beyond keyword matching
- Handles synonyms and related concepts naturally

### Modular Knowledge
- Easy to add new knowledge capsules
- Automatic connection creation based on similarity
- Categorized organization (health, technology, education, etc.)

### Emergent Behavior
- Complex responses emerge from simple capsule interactions
- Context-aware activation patterns
- Synergistic knowledge combination

## 🔧 Configuration

### Algorithm Parameters

```python
# In KnowledgeCapsulesAlgorithm.__init__()
self.activation_threshold = 0.4  # Capsule activation threshold
self.learning_rate = 0.1         # Learning speed for feedback
```

### Encoder Settings

```python
# In EmbeddedNeo4jConnection.__init__()
self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
# Alternative models:
# - 'all-mpnet-base-v2' (better quality, slower)
# - 'all-distilroberta-v1' (faster, good performance)
```

### Connection Thresholds

```python
# In _auto_create_connections()
if similarity > 0.3:  # Auto-connection threshold
    # Create bidirectional connections
```

## 📁 File Structure

```
knowledge-capsules/
│
├── capsules_app.py              # Main application interface
├── capsules_algorithm.py        # Core 5-step processing algorithm
├── embedded_neo4j_setup.py      # Database and vector operations
├── capsules_db/                 # Local database storage
│   └── capsules_data.pkl        # Serialized capsules and connections
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧪 Advanced Usage

### Custom Capsule Categories

```python
# Add domain-specific knowledge
app.add_custom_capsule("machine_learning", 
    "Machine learning algorithms learn patterns from data to make predictions",
    "ai")

app.add_custom_capsule("deep_learning",
    "Deep learning uses neural networks with multiple layers for complex pattern recognition", 
    "ai")
```

### Batch Knowledge Loading

```python
knowledge_base = [
    ("topic1", "description1", "category1"),
    ("topic2", "description2", "category2"),
    # ... more capsules
]

for capsule_id, text, category in knowledge_base:
    app.add_custom_capsule(capsule_id, text, category)
```

### Fine-tuning Parameters

```python
# Adjust for different use cases
algorithm = KnowledgeCapsulesAlgorithm()

# For strict matching (technical domains)
algorithm.activation_threshold = 0.6

# For broad matching (creative domains)  
algorithm.activation_threshold = 0.3

# Faster learning
algorithm.learning_rate = 0.2

# Conservative learning
algorithm.learning_rate = 0.05
```

## 🤖 Use Cases

### Educational Systems
- Personalized learning recommendations
- Adaptive Q&A for students
- Knowledge gap identification

### Customer Support
- Intelligent help desk responses
- Context-aware troubleshooting
- Escalation pattern learning

### Research Assistance
- Literature connection discovery
- Concept relationship mapping
- Hypothesis generation support

### Personal Knowledge Management
- Note organization and retrieval
- Idea connection and development
- Memory augmentation

## 🔮 Future Enhancements

### Planned Features
- **Multi-modal Capsules**: Support for images, audio, video
- **Hierarchical Capsules**: Sub-capsule and super-capsule relationships
- **Temporal Dynamics**: Time-aware activation and decay
- **Distributed Processing**: Multi-agent capsule networks

### Integration Possibilities
- **API Endpoints**: REST API for web applications
- **Database Backends**: PostgreSQL, MongoDB integration
- **Cloud Deployment**: AWS, Azure, GCP support
- **Real-time Learning**: Streaming feedback incorporation

## 🐛 Troubleshooting

### Common Issues

**Issue**: Slow initial startup
- **Solution**: First-time model download (~90MB). Subsequent runs are fast.

**Issue**: Poor response quality
- **Solution**: Add more relevant capsules, adjust activation threshold, provide feedback

**Issue**: Memory usage high
- **Solution**: Implement capsule pruning, use smaller encoder model

**Issue**: Connections not forming
- **Solution**: Lower similarity threshold in `_auto_create_connections()`

## 📚 References

- **Sentence Transformers**: https://www.sbert.net/
- **Capsule Networks**: Sabour et al., "Dynamic Routing Between Capsules" (2017)
- **Neo4j Graph Database**: https://neo4j.com/
- **Cosine Similarity**: Vector space information retrieval

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Sentence Transformers team for semantic encoding models
- Neo4j team for graph database inspiration
- Research community for capsule network concepts

---

*Built with ❤️ and lots of ☕*