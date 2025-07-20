"""
Updated Embedded Neo4j Setup - Uses external modular encoder
Cleaner separation of concerns
"""

import os
import numpy as np
from typing import List, Dict
from modular_encoder import ModularEncoder

class EmbeddedNeo4jConnection:
    """
    Updated Neo4j connection that uses external encoder
    Focuses only on data storage and graph operations
    """
    
    def __init__(self, db_path="./capsules_db", encoder=None):
        """Initialize with external encoder"""
        self.db_path = os.path.abspath(db_path)
        
        # Use provided encoder or create new one
        self.encoder = encoder if encoder else ModularEncoder()
        
        # Create database directory
        os.makedirs(self.db_path, exist_ok=True)
        
        # In-memory graph structure
        self.capsules = {}
        self.connections = {}
        
        print(f"✅ Database initialized at: {self.db_path}")
    
    def create_capsule(self, capsule_id: str, text_content: str, category: str = "general"):
        """Create Knowledge Capsule with vector embedding"""
        print(f"📦 Creating capsule: {capsule_id}")
        
        # Generate vector using external encoder
        vector = self.encoder.encode(text_content)
        
        # Store capsule data
        self.capsules[capsule_id] = {
            'id': capsule_id,
            'text': text_content,
            'vector': vector,
            'category': category,
            'activation': 0.0,
            'is_active': False
        }
        
        # Auto-create connections
        self._auto_create_connections(capsule_id)
        print(f"✅ Capsule '{capsule_id}' created")
    
    def _auto_create_connections(self, new_capsule_id: str):
        """Create connections based on similarity"""
        if new_capsule_id not in self.connections:
            self.connections[new_capsule_id] = []
        
        for existing_id, existing_data in self.capsules.items():
            if existing_id == new_capsule_id:
                continue
            
            # Use encoder's similarity calculation
            similarity = self.encoder.calculate_similarity(
                self.capsules[new_capsule_id]['text'], 
                existing_data['vector']
            )
            
            if similarity > 0.3:
                # Bidirectional connections
                self.connections[new_capsule_id].append({
                    'connected_id': existing_id,
                    'weight': float(similarity)
                })
                
                if existing_id not in self.connections:
                    self.connections[existing_id] = []
                
                if new_capsule_id not in [c['connected_id'] for c in self.connections[existing_id]]:
                    self.connections[existing_id].append({
                        'connected_id': new_capsule_id,
                        'weight': float(similarity)
                    })
    
    def get_all_capsules(self) -> List[Dict]:
        """Get all capsules"""
        return list(self.capsules.values())
    
    def get_capsule_connections(self, capsule_id: str) -> List[Dict]:
        """Get connections for capsule"""
        return self.connections.get(capsule_id, [])
    
    def update_capsule_activation(self, capsule_id: str, activation: float, is_active: bool):
        """Update activation state"""
        if capsule_id in self.capsules:
            self.capsules[capsule_id]['activation'] = activation
            self.capsules[capsule_id]['is_active'] = is_active
    
    def calculate_similarity(self, query_text: str, capsule_vector) -> float:
        """Calculate similarity using external encoder"""
        return self.encoder.calculate_similarity(query_text, capsule_vector)
    
    def update_connection_weight(self, capsule1_id: str, capsule2_id: str, new_weight: float):
        """Update connection weight"""
        if capsule1_id in self.connections:
            for conn in self.connections[capsule1_id]:
                if conn['connected_id'] == capsule2_id:
                    conn['weight'] = new_weight
        
        if capsule2_id in self.connections:
            for conn in self.connections[capsule2_id]:
                if conn['connected_id'] == capsule1_id:
                    conn['weight'] = new_weight
    
    def save_database(self):
        """Save to disk"""
        import pickle
        data = {'capsules': self.capsules, 'connections': self.connections}
        with open(os.path.join(self.db_path, 'capsules_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print("💾 Database saved")
    
    def load_database(self):
        """Load from disk"""
        import pickle
        data_file = os.path.join(self.db_path, 'capsules_data.pkl')
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            self.capsules = data['capsules']
            self.connections = data['connections']
            print("📂 Database loaded")
        else:
            print("📂 Starting fresh database")
    
    def close(self):
        """Close and save"""
        self.save_database()
        print("✅ Database closed")
