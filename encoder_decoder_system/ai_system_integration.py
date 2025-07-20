"""
AI System Integration - Complete Modular Architecture
Encoder → Knowledge Capsules → Transformer Decoder
"""

import numpy as np
from typing import List, Dict, Tuple
from modular_encoder import ModularEncoder
from updated_capsules_algorithm import KnowledgeCapsulesAlgorithm
from transformer_decoder import TransformerDecoder, MinimalDecoder

class ModularAISystem:
    """
    Complete AI system with separate encoder and decoder transformers
    Clean modular architecture for future upgrades
    """
    
    def __init__(self, db_path="./capsules_db", use_transformer_decoder=True):
        """Initialize complete modular system"""
        print("🌟 Initializing Modular AI System...")
        print("=" * 50)
        
        # Initialize components separately
        print("1️⃣ Loading encoder...")
        self.encoder = ModularEncoder()
        
        print("2️⃣ Loading knowledge capsules...")
        self.knowledge_system = KnowledgeCapsulesAlgorithm(db_path)
        
        print("3️⃣ Loading decoder...")
        if use_transformer_decoder:
            try:
                self.decoder = TransformerDecoder()
            except Exception:
                print("⚠️ Transformer decoder failed, using minimal decoder")
                self.decoder = MinimalDecoder()
        else:
            self.decoder = MinimalDecoder()
        
        print("✅ Modular AI System ready!")
        print("=" * 50)
    
    def process_query(self, query: str) -> Tuple[str, Dict]:
        """
        Complete processing pipeline:
        Query → Encoder → Knowledge Capsules (with vectors) → Decoder → Response
        """
        print(f"\n🔄 Processing: '{query}'")
        
        # Step 1: Encode query
        query_vector = self.encoder.encode(query)
        print(f"📊 Query encoded to {len(query_vector)} dimensions")
        
        # Step 2: Process through knowledge capsules (returns capsule response + active IDs)
        capsule_response, active_capsule_ids = self.knowledge_system.process_query(query)
        
        # Step 3: Get full capsule data with vectors for decoder
        all_capsules = self.knowledge_system.db.get_all_capsules()
        active_capsules = [c for c in all_capsules if c['id'] in active_capsule_ids]
        
        print(f"🧠 Knowledge system activated {len(active_capsules)} capsules")
        
        # Step 4: Pass query, active capsules (with vectors), and query vector to decoder
        final_response = self.decoder.decode_response(query, active_capsules, query_vector)
        
        print("🤖 Final response generated through decoder using vector embeddings")
        
        # Prepare metadata
        metadata = {
            'query_vector_norm': float(np.linalg.norm(query_vector)),
            'active_capsules': len(active_capsules),
            'capsule_ids': active_capsule_ids,
            'capsule_vectors': [c['vector'].tolist()[:5] for c in active_capsules][:3],  # First 5 dims of top 3 capsules
            'original_capsule_response': capsule_response,
            'encoder_model': type(self.encoder).__name__,
            'decoder_model': type(self.decoder).__name__
        }
        
        return final_response, metadata
    
    def learn_from_feedback(self, feedback: str):
        """Forward feedback to knowledge system"""
        print(f"📝 Learning from feedback: '{feedback}'")
        self.knowledge_system.learn_from_feedback(feedback)
        print("✅ Feedback processed")
    
    def add_knowledge(self, capsule_id: str, text: str, category: str = "general"):
        """Add new knowledge capsule"""
        print(f"➕ Adding knowledge: {capsule_id}")
        self.knowledge_system.add_capsule(capsule_id, text, category)
        print("✅ Knowledge added")
    
    def get_system_status(self) -> Dict:
        """Get current system status with vector information"""
        all_capsules = self.knowledge_system.db.get_all_capsules()
        active_capsules = [c for c in all_capsules if c.get('is_active', False)]
        
        # Calculate vector statistics
        if all_capsules:
            sample_vector = all_capsules[0]['vector']
            vector_dim = len(sample_vector)
            avg_activation = sum(c.get('activation', 0) for c in active_capsules) / len(active_capsules) if active_capsules else 0
        else:
            vector_dim = 0
            avg_activation = 0
        
        status = {
            'total_capsules': len(all_capsules),
            'active_capsules': len(active_capsules),
            'encoder_dimensions': self.encoder.embedding_dim,
            'capsule_vector_dimensions': vector_dim,
            'average_activation': round(avg_activation, 3),
            'decoder_type': type(self.decoder).__name__
        }
        
        print(f"📊 System Status: {status}")
        return status
    
    def show_active_knowledge(self):
        """Display currently active knowledge"""
        self.knowledge_system.show_status()
    
    def save_system(self):
        """Save system state"""
        print("💾 Saving system...")
        self.knowledge_system.db.save_database()
        print("✅ System saved")
    
    def close(self):
        """Properly close system"""
        print("🔄 Closing system...")
        self.knowledge_system.close()
        print("✅ System closed")

# Example usage and testing
def main():
    """Example usage of the modular AI system"""
    print("🚀 Starting AI System Demo")
    
    # Initialize system
    ai = ModularAISystem()
    
    # Add some knowledge
    ai.add_knowledge("health_001", "Regular exercise improves cardiovascular health and mental well-being", "health")
    ai.add_knowledge("tech_001", "Python is a versatile programming language used for AI and web development", "technology")
    ai.add_knowledge("edu_001", "Active learning techniques improve knowledge retention significantly", "education")
    
    # Create connections between related capsules
    ai.add_knowledge("health_002", "Mental health is improved through physical exercise and proper nutrition", "health")
    
    # Test queries
    test_queries = [
        "How can I improve my health?",
        "What is Python used for?",
        "How to learn effectively?",
        "Tell me about exercise benefits"
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        response, metadata = ai.process_query(query)
        print(f"💬 Response: {response}")
        print(f"📊 Metadata: {metadata}")
        print(f"🔢 Vector Info: Query norm: {metadata['query_vector_norm']:.3f}, Capsule vectors included: {len(metadata.get('capsule_vectors', []))}")
        
        # Simulate feedback
        ai.learn_from_feedback("good response")
    
    # Show system status
    ai.get_system_status()
    ai.show_active_knowledge()
    
    # Save and close
    ai.save_system()
    ai.close()
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()