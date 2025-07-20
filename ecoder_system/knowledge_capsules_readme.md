# 🧠 Knowledge Capsules Algorithm

An intelligent Q&A system that uses interconnected knowledge capsules to provide contextual answers through a 5-step neural-inspired algorithm.

## 🌟 Overview

The Knowledge Capsules Algorithm mimics how human knowledge works - through interconnected concepts that activate and reinforce each other. When you ask a question, the system:

1. **Calculates similarities** between your query and all knowledge capsules
2. **Applies activation function** to determine which capsules are relevant
3. **Calculates inter-capsule signals** to boost related knowledge
4. **Determines final activations** combining original similarity + signal boosts
5. **Generates intelligent responses** from the most active capsules

## 🏗️ Architecture

```
Query → Similarity → Activation → Signals → Final → Response
         Calc        Function    Calc      Activation
```

### Core Components

- **Knowledge Capsules**: Self-contained units of knowledge with vector embeddings
- **Connection Network**: Weighted connections between related capsules
- **Activation System**: Threshold-based activation with signal propagation
- **Learning Mechanism**: Feedback-based connection strengthening/weakening

## 📦 Installation

### Prerequisites
```bash
pip install neo4j sentence-transformers scikit-learn numpy
```

### Files Structure
```
knowledge-capsules/
├── capsules_app.py              # Main application
├── capsules_algorithm.py        # Core algorithm
├── embedded_neo4j_setup.py      # Database setup
└── README.md                    # This file
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

# Provide feedback to improve the system
app.give_feedback("good")  # or "bad"

# Check system status
app.show_system_status()
```

### Interactive Session

```python
# Run the main application
python capsules_app.py
```

This will:
1. Initialize the system with pre-built knowledge capsules
2. Run demo questions
3. Offer an interactive session where you can ask questions

## 💡 How It Works

### 1. Knowledge Capsules

Each capsule contains:
- **Text content**: The actual knowledge
- **Vector embedding**: Mathematical representation for similarity
- **Category**: Organization (health, technology, education)
- **Activation level**: Current activation strength
- **Connections**: Links to related capsules

```python
# Example capsule
{
    'id': 'exercise',
    'text': 'Regular physical exercise strengthens muscles...',
    'vector': [0.1, -0.3, 0.8, ...],  # 384-dim vector
    'category': 'health',
    'activation': 0.85,
    'is_active': True
}
```

### 2. Connection Network

Capsules automatically connect based on semantic similarity:
- **Auto-connection**: Similarity > 0.3 threshold
- **Bidirectional**: Connections work both ways
- **Weighted**: Connection strength varies (0.0 to 1.0)
- **Learning**: Weights adjust based on feedback

### 3. The 5-Step Algorithm

#### Step 1: Calculate Similarities
```python
similarity = cosine_similarity(query_vector, capsule_vector)
```

#### Step 2: Apply Activation Function
```python
activation = similarity if similarity > threshold else 0.0
```

#### Step 3: Calculate Inter-Capsule Signals
```python
signal = active_capsule_activation * connection_weight
```

#### Step 4: Final Activations
```python
final_activation = original_similarity + received_signals
```

#### Step 5: Generate Response
- Select top 3 most active capsules
- Combine their knowledge with confidence scores
- Add contextual advice based on query type

## 🎯 Usage Examples

### Example 1: Health Query

```python
app = KnowledgeCapsulesApp()
response = app.ask_question("I want to stay healthy, what should I do?")
```

**Output:**
```
📊 Calculating similarities...
  exercise: 0.789
  nutrition: 0.751
  sleep: 0.623
  hydration: 0.234

⚡ Applying activation function...
  ✅ exercise: ACTIVE (0.789)
  ✅ nutrition: ACTIVE (0.751)
  ✅ sleep: ACTIVE (0.623)
  💤 hydration: inactive (0.234)

🔊 Calculating capsule signals...
  📡 exercise → nutrition: 0.142
  📡 exercise → sleep: 0.118
  📡 nutrition → exercise: 0.135

💡 Final activations...
  exercise: 0.789 + 0.135 = 0.924
  nutrition: 0.751 + 0.142 = 0.893
  sleep: 0.623 + 0.118 = 0.741

🤖 Response:
Based on my knowledge: Regular physical exercise strengthens muscles, improves heart health, and boosts mood (0.92) | Balanced nutrition with fruits, vegetables, proteins provides essential vitamins (0.89) | Quality sleep of 7-9 hours allows body repair and memory consolidation (0.74)

🔥 Active Capsules: [exercise, nutrition, sleep]
```

### Example 2: Learning Query

```python
response = app.ask_question("How can I learn programming effectively?")
```

This would activate programming, effective learning, and time management capsules.

### Example 3: Adding Custom Knowledge

```python
app.add_custom_capsule(
    capsule_id="meditation",
    knowledge_text="Meditation reduces stress, improves focus, and enhances emotional well-being through mindfulness practice",
    category="health"
)
```

## 🧪 Advanced Features

### Learning from Feedback

```python
# After asking a question
app.ask_question("How to reduce stress?")

# Provide feedback
app.give_feedback("good")  # Strengthens active connections
app.give_feedback("bad")   # Weakens active connections
```

### System Monitoring

```python
app.show_system_status()
```

**Output:**
```
📊 SYSTEM STATUS:
Total capsules: 11
Active capsules: 3

🔥 Currently active:
  • exercise: 0.924
  • nutrition: 0.893
  • sleep: 0.741

🔗 Active capsule connections:
  exercise:
    → nutrition (strength: 0.678)
    → sleep (strength: 0.542)
    → stress_management (strength: 0.445)
```

### Custom Algorithm Parameters

```python
from capsules_algorithm import KnowledgeCapsulesAlgorithm

# Custom configuration
algorithm = KnowledgeCapsulesAlgorithm()
algorithm.activation_threshold = 0.5  # Higher threshold
algorithm.learning_rate = 0.2         # Faster learning
```

## 📊 Built-in Knowledge Domains

The system comes pre-loaded with knowledge in three domains:

### 🏥 Health & Wellness
- Exercise and physical fitness
- Nutrition and balanced diet
- Sleep and rest
- Hydration
- Stress management

### 💻 Technology
- Programming fundamentals
- AI basics
- Web development

### 📚 Learning & Education
- Effective learning techniques
- Time management
- Memory improvement methods

## 🔧 Configuration Options

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `activation_threshold` | 0.4 | Minimum similarity for capsule activation |
| `learning_rate` | 0.1 | Rate of connection weight updates |
| `connection_threshold` | 0.3 | Minimum similarity for auto-connections |

### Vector Model

The system uses `all-MiniLM-L6-v2` from Sentence Transformers:
- **Dimensions**: 384
- **Speed**: Fast inference
- **Quality**: Good semantic understanding

To change the model:
```python
# In embedded_neo4j_setup.py
self.encoder = SentenceTransformer('your-model-name')
```

## 🎮 Interactive Commands

When running the interactive session:

| Command | Action |
|---------|--------|
| `quit` / `exit` / `q` | Exit the session |
| `status` | Show system status |
| Any question | Process and respond |
| `good` / `bad` | Provide feedback after response |
| `skip` | Skip feedback |

## 🔬 Technical Details

### Vector Similarity
- **Method**: Cosine similarity
- **Range**: -1.0 to 1.0
- **Threshold**: 0.4 for activation, 0.3 for connections

### Signal Propagation
```python
signal_strength = sender_activation * connection_weight
final_activation = original_similarity + sum(received_signals)
```

### Learning Algorithm
- **Positive feedback**: Increases connection weights by learning_rate
- **Negative feedback**: Decreases connection weights by learning_rate
- **Bounds**: Weights clamped between 0.0 and 1.0

### Database Storage
- **Format**: Pickle files (simple implementation)
- **Location**: `./capsules_db/`
- **Auto-save**: On application close
- **Auto-load**: On application start

## 🚧 Limitations & Future Improvements

### Current Limitations
- Simple in-memory database (not production Neo4j)
- Limited to pre-defined knowledge domains
- No real-time knowledge updates
- Basic feedback mechanism

### Planned Improvements
- Full Neo4j integration
- Dynamic knowledge acquisition
- More sophisticated learning algorithms
- Web interface
- Multi-modal knowledge (text, images, etc.)
- Contextual conversation memory

## 🤝 Contributing

1. **Add new knowledge domains**:
   ```python
   app.add_custom_capsule("domain_concept", "Knowledge text...", "domain")
   ```

2. **Improve the algorithm**:
   - Experiment with different activation functions
   - Try alternative similarity metrics
   - Implement more sophisticated learning

3. **Extend functionality**:
   - Add new response generation strategies
   - Implement conversation context
   - Create visualization tools

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🆘 Troubleshooting

### Common Issues

**Q: "No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**Q: "Low similarity scores for expected matches"**
- Check if your query uses similar vocabulary to the capsule content
- Consider adding more specific capsules
- Adjust the activation threshold

**Q: "No capsules activating"**
- Lower the activation_threshold
- Add more relevant knowledge capsules
- Check if the vector model is loaded correctly

**Q: "Database not saving/loading"**
- Ensure write permissions in the capsules_db directory
- Check available disk space
- Verify pickle module is available

**Q: "Poor response quality"**
- Add more detailed knowledge to capsules
- Increase the number of capsules in relevant domains
- Provide feedback to improve connections
- Consider adjusting the learning rate

### Debug Mode

Enable verbose logging by modifying the print statements or adding:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Metrics

### Typical Performance
- **Capsule creation**: ~50ms per capsule
- **Query processing**: ~100-300ms
- **Memory usage**: ~50MB with 50 capsules
- **Similarity calculation**: O(n) where n = number of capsules

### Optimization Tips
1. **Batch capsule creation** for better performance
2. **Cache vector embeddings** for repeated queries
3. **Limit active connections** to top-k most relevant
4. **Use smaller vector models** for faster inference

## 🔍 Example Workflows

### Workflow 1: Building a Personal Knowledge Assistant

```python
# 1. Initialize system
app = KnowledgeCapsulesApp()

# 2. Add personal knowledge
app.add_custom_capsule("python_tips", 
    "Use list comprehensions for concise loops, leverage generators for memory efficiency", 
    "programming")

app.add_custom_capsule("work_productivity", 
    "Use time blocking, minimize context switching, take regular breaks", 
    "productivity")

# 3. Ask domain-specific questions
app.ask_question("How can I be more productive at work?")
app.give_feedback("good")

app.ask_question("What are some Python best practices?")
app.give_feedback("good")

# 4. Monitor learning
app.show_system_status()
```

### Workflow 2: Educational Content System

```python
# Create subject-specific capsules
subjects = [
    ("algebra_basics", "Algebra involves solving equations with variables like x and y", "math"),
    ("geometry_concepts", "Geometry studies shapes, angles, and spatial relationships", "math"),
    ("physics_motion", "Motion involves displacement, velocity, and acceleration over time", "science")
]

for capsule_id, text, category in subjects:
    app.add_custom_capsule(capsule_id, text, category)

# Student asks questions
app.ask_question("How do I solve for x in equations?")
app.ask_question("What is acceleration?")
```

### Workflow 3: Troubleshooting Assistant

```python
# Technical troubleshooting knowledge
troubleshooting = [
    ("network_issues", "Check cable connections, restart router, verify IP configuration", "tech_support"),
    ("software_crashes", "Update drivers, check system requirements, scan for malware", "tech_support"),
    ("performance_slow", "Close unnecessary programs, clear temporary files, add more RAM", "tech_support")
]

for capsule_id, text, category in troubleshooting:
    app.add_custom_capsule(capsule_id, text, category)

# User reports issues
app.ask_question("My computer is running very slowly")
app.ask_question("I can't connect to the internet")
```

## 🎨 Customization Examples

### Custom Activation Function

```python
# Modify _apply_activation in capsules_algorithm.py
def _apply_activation(self, similarities: Dict[str, float]) -> Dict[str, float]:
    """Custom sigmoid activation function"""
    import math
    activations = {}
    
    for capsule_id, similarity in similarities.items():
        # Sigmoid activation instead of threshold
        activation = 1 / (1 + math.exp(-10 * (similarity - 0.5)))
        activations[capsule_id] = activation
    
    return activations
```

### Custom Response Generation

```python
# Modify _generate_response for domain-specific responses
def _generate_response(self, query: str, active_capsules: List[Dict]) -> str:
    if not active_capsules:
        return "I don't have enough relevant knowledge to answer that question."
    
    # Domain-specific response formatting
    health_capsules = [c for c in active_capsules if c['category'] == 'health']
    tech_capsules = [c for c in active_capsules if c['category'] == 'technology']
    
    response = ""
    if health_capsules:
        response += "🏥 Health Advice: " + " | ".join([c['text'] for c in health_capsules[:2]])
    
    if tech_capsules:
        if response: response += "\n\n"
        response += "💻 Technical Info: " + " | ".join([c['text'] for c in tech_capsules[:2]])
    
    return response
```

### Custom Connection Logic

```python
# Modify _auto_create_connections for specialized connections
def _auto_create_connections(self, new_capsule_id: str):
    """Create connections based on custom rules"""
    new_capsule = self.capsules[new_capsule_id]
    
    for existing_id, existing_data in self.capsules.items():
        if existing_id == new_capsule_id:
            continue
        
        # Category-based connections
        if new_capsule['category'] == existing_data['category']:
            connection_weight = 0.8  # Strong intra-category connection
        else:
            # Calculate semantic similarity
            similarity = cosine_similarity([new_capsule['vector']], [existing_data['vector']])[0][0]
            connection_weight = max(0, similarity - 0.1)  # Slight bias reduction
        
        if connection_weight > 0.3:
            self._create_bidirectional_connection(new_capsule_id, existing_id, connection_weight)
```

## 📚 Research Applications

### Academic Research Support

```python
# Research paper knowledge base
research_capsules = [
    ("ml_supervised", "Supervised learning uses labeled data to train models for prediction", "research"),
    ("ml_unsupervised", "Unsupervised learning finds patterns in unlabeled data", "research"),
    ("deep_learning", "Deep learning uses neural networks with multiple layers", "research"),
    ("nlp_transformers", "Transformers use attention mechanisms for sequence processing", "research")
]

# Researcher queries
app.ask_question("What are the main types of machine learning?")
app.ask_question("How do transformers work in NLP?")
```

### Business Knowledge Management

```python
# Company-specific knowledge
business_capsules = [
    ("sales_process", "Our sales process involves lead qualification, demo, proposal, closing", "business"),
    ("customer_support", "Customer support follows ticket triage, technical resolution, follow-up", "business"),
    ("product_features", "Our product offers real-time analytics, custom dashboards, API integration", "business")
]

# Employee queries
app.ask_question("What is our standard sales process?")
app.ask_question("How should I handle customer support tickets?")
```

## 🎯 Best Practices

### Knowledge Capsule Design
1. **Be Specific**: Each capsule should contain focused, specific knowledge
2. **Use Clear Language**: Avoid jargon unless domain-appropriate
3. **Optimal Length**: 1-3 sentences per capsule works best
4. **Consistent Terminology**: Use consistent vocabulary across related capsules
5. **Regular Updates**: Keep knowledge current and accurate

### System Configuration
1. **Start Conservative**: Begin with higher thresholds, lower as needed
2. **Monitor Performance**: Check system status regularly
3. **Provide Feedback**: Consistent feedback improves system performance
4. **Batch Operations**: Add multiple related capsules together
5. **Test Thoroughly**: Validate responses before deployment

### Query Optimization
1. **Use Natural Language**: System works best with natural queries
2. **Be Specific**: Specific questions get better responses
3. **Iterate**: Refine queries based on initial responses
4. **Provide Context**: Include relevant context in questions
5. **Give Feedback**: Always provide feedback for learning

## 🔮 Future Vision

The Knowledge Capsules Algorithm represents a step toward more human-like AI reasoning, where:

- **Knowledge is Connected**: Like human memory, information connects naturally
- **Context Matters**: Responses consider related knowledge automatically
- **Learning is Continuous**: System improves through interaction
- **Reasoning is Transparent**: Users can see why certain answers were given

### Potential Applications
- Personal knowledge assistants
- Educational tutoring systems
- Enterprise knowledge management
- Research collaboration tools
- Decision support systems

## 📞 Support & Community

For questions, suggestions, or contributions:

1. **Issues**: Report bugs or request features
2. **Discussions**: Share use cases and improvements
3. **Documentation**: Help improve this README
4. **Code**: Contribute new features or optimizations

---

**Made with ❤️ for the AI community**

*"Knowledge is not just what you know, but how your thoughts connect."*