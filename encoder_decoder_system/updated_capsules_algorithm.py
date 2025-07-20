"""
Updated Knowledge Capsules Algorithm - Uses modular encoder
"""

from updated_neo4j_setup import EmbeddedNeo4jConnection
from modular_encoder import ModularEncoder
from typing import List, Dict, Tuple

class KnowledgeCapsulesAlgorithm:
    """Updated algorithm with modular encoder"""
    
    def __init__(self, db_path="./capsules_db"):
        """Initialize with shared encoder"""
        # Create shared encoder
        self.encoder = ModularEncoder()
        
        # Initialize database with shared encoder
        self.db = EmbeddedNeo4jConnection(db_path, self.encoder)
        self.db.load_database()
        
        # Algorithm parameters
        self.activation_threshold = 0.4
        self.learning_rate = 0.1
        self.last_active_capsules = []
        
        print("🚀 Knowledge Capsules Algorithm initialized")
    
    def process_query(self, question: str) -> Tuple[str, List[str]]:
        """Main processing with 5-step algorithm"""
        print(f"\n🤔 Processing: '{question}'")
        
        # Step 1: Calculate similarities
        similarities = self._calculate_similarities(question)
        
        # Step 2: Apply activation function  
        initial_activations = self._apply_activation(similarities)
        
        # Step 3: Calculate capsule signals
        signal_boosts = self._calculate_signals(initial_activations)
        
        # Step 4: Calculate final activations
        final_activations = self._calculate_final_activations(similarities, signal_boosts)
        
        # Step 5: Generate response
        active_capsules = self._get_active_capsules(final_activations)
        response = self._generate_response(question, active_capsules)
        
        self.last_active_capsules = active_capsules
        return response, [c['id'] for c in active_capsules]
    
    def _calculate_similarities(self, query: str) -> Dict[str, float]:
        """Step 1: Calculate similarities using modular encoder"""
        similarities = {}
        capsules = self.db.get_all_capsules()
        
        print("📊 Calculating similarities...")
        for capsule in capsules:
            similarity = self.encoder.calculate_similarity(query, capsule['vector'])
            similarities[capsule['id']] = similarity
            print(f"  {capsule['id']}: {similarity:.3f}")
        
        return similarities
    
    def _apply_activation(self, similarities: Dict[str, float]) -> Dict[str, float]:
        """Step 2: Apply activation function"""
        activations = {}
        print("⚡ Applying activation...")
        for capsule_id, similarity in similarities.items():
            if similarity > self.activation_threshold:
                activations[capsule_id] = similarity
                print(f"  ✅ {capsule_id}: ACTIVE ({similarity:.3f})")
            else:
                activations[capsule_id] = 0.0
                print(f"  💤 {capsule_id}: inactive ({similarity:.3f})")
        return activations
    
    def _calculate_signals(self, activations: Dict[str, float]) -> Dict[str, float]:
        """Step 3: Calculate inter-capsule signals"""
        signals = {cid: 0.0 for cid in activations.keys()}
        print("🔊 Calculating signals...")
        
        for capsule_id, activation in activations.items():
            if activation > 0:
                connections = self.db.get_capsule_connections(capsule_id)
                for conn in connections:
                    connected_id = conn['connected_id']
                    weight = conn['weight']
                    signal = activation * weight
                    
                    if connected_id in signals:
                        signals[connected_id] += signal
                        print(f"  📡 {capsule_id} → {connected_id}: {signal:.3f}")
        return signals
    
    def _calculate_final_activations(self, similarities: Dict[str, float], 
                                   signals: Dict[str, float]) -> Dict[str, float]:
        """Step 4: Final activations"""
        final_activations = {}
        print("💡 Final activations...")
        
        for capsule_id in similarities.keys():
            original = similarities[capsule_id]
            boost = signals.get(capsule_id, 0.0)
            final = original + boost
            final_activations[capsule_id] = final
            print(f"  🧮 {capsule_id}: {original:.3f} + {boost:.3f} = {final:.3f}")
        
        return final_activations
    
    def _get_active_capsules(self, final_activations: Dict[str, float]) -> List[Dict]:
        """Step 5a: Get active capsules"""
        active_capsules = []
        print("🔥 Active capsules:")
        
        for capsule_id, activation in final_activations.items():
            is_active = activation > self.activation_threshold
            self.db.update_capsule_activation(capsule_id, activation, is_active)
            
            if is_active:
                capsule_data = self.db.capsules[capsule_id].copy()
                capsule_data['activation'] = activation
                active_capsules.append(capsule_data)
                print(f"  ✅ {capsule_id}: {activation:.3f}")
        
        active_capsules.sort(key=lambda x: x['activation'], reverse=True)
        return active_capsules
    
    def _generate_response(self, query: str, active_capsules: List[Dict]) -> str:
        """Step 5b: Generate response"""
        if not active_capsules:
            return "I don't have relevant knowledge for that question."
        
        top_capsules = active_capsules[:3]
        knowledge_parts = []
        
        for capsule in top_capsules:
            knowledge_parts.append(f"{capsule['text']} (confidence: {capsule['activation']:.2f})")
        
        response = "Based on knowledge: " + " | ".join(knowledge_parts)
        
        # Simple contextual additions
        query_lower = query.lower()
        if any(word in query_lower for word in ['how', 'what should']):
            response += "\n\nApply this to your situation."
        elif 'why' in query_lower:
            response += "\n\nThis explains the reasoning."
        
        return response
    
    def learn_from_feedback(self, feedback: str):
        """Learning from feedback"""
        if not self.last_active_capsules:
            print("⚠️ No recent query")
            return
        
        feedback_lower = feedback.lower()
        if any(word in feedback_lower for word in ['good', 'correct', 'yes']):
            self._strengthen_connections()
            print("📈 Strengthened connections")
        elif any(word in feedback_lower for word in ['bad', 'wrong', 'no']):
            self._weaken_connections()
            print("📉 Weakened connections")
    
    def _strengthen_connections(self):
        """Strengthen connections"""
        active_ids = [c['id'] for c in self.last_active_capsules]
        for capsule_id in active_ids:
            connections = self.db.get_capsule_connections(capsule_id)
            for conn in connections:
                if conn['connected_id'] in active_ids:
                    old_weight = conn['weight']
                    new_weight = min(1.0, old_weight + self.learning_rate)
                    self.db.update_connection_weight(capsule_id, conn['connected_id'], new_weight)
    
    def _weaken_connections(self):
        """Weaken connections"""
        active_ids = [c['id'] for c in self.last_active_capsules]
        for capsule_id in active_ids:
            connections = self.db.get_capsule_connections(capsule_id)
            for conn in connections:
                if conn['connected_id'] in active_ids:
                    old_weight = conn['weight']
                    new_weight = max(0.0, old_weight - self.learning_rate)
                    self.db.update_connection_weight(capsule_id, conn['connected_id'], new_weight)
    
    def add_capsule(self, capsule_id: str, text: str, category: str = "general"):
        """Add new capsule"""
        self.db.create_capsule(capsule_id, text, category)
        print(f"✅ Added: {capsule_id}")
    
    def show_status(self):
        """Show status"""
        capsules = self.db.get_all_capsules()
        active = [c for c in capsules if c['is_active']]
        
        print(f"\n📊 STATUS: {len(capsules)} total, {len(active)} active")
        if active:
            print("🔥 Active:")
            for capsule in active:
                print(f"  • {capsule['id']}: {capsule['activation']:.3f}")
    
    def close(self):
        """Close system"""
        self.db.close()
