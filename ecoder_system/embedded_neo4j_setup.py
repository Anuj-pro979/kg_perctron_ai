"""
Embedded Neo4j Setup for Knowledge Capsules
Uses local embedded database - no external server needed
"""

import os
from neo4j import GraphDatabase  # Removed unused import
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class EmbeddedNeo4jConnection:
    """
    Embedded Neo4j database for Knowledge Capsules
    Stores everything locally in ./capsules_db/ folder
    """
    
    def __init__(self, db_path="./capsules_db"):
        """Initialize embedded Neo4j database"""
        self.db_path = os.path.abspath(db_path)
        self.uri = "neo4j://localhost:7687"  # Will use embedded if available
        
        # Initialize encoder for vector embeddings
        print("🧠 Loading sentence encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create database directory
        os.makedirs(self.db_path, exist_ok=True)
        
        # For simplicity, use in-memory graph structure
        # In production, you'd use actual embedded Neo4j
        self.capsules = {}  # {id: capsule_data}
        self.connections = {}  # {id: [connected_ids]}
        
        print(f"✅ Database initialized at: {self.db_path}")
    
    def create_capsule(self, capsule_id: str, text_content: str, category: str = "general"):
        """Create a Knowledge Capsule with vector embedding"""
        print(f"📦 Creating capsule: {capsule_id}")
        
        # Generate vector embedding
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
        
        # Auto-create connections with existing capsules
        self._auto_create_connections(capsule_id)
        
        print(f"✅ Capsule '{capsule_id}' created and connected")
    
    def _auto_create_connections(self, new_capsule_id: str):
        """Automatically create connections based on vector similarity"""
        if new_capsule_id not in self.connections:
            self.connections[new_capsule_id] = []
        
        new_vector = self.capsules[new_capsule_id]['vector']
        
        for existing_id, existing_data in self.capsules.items():
            if existing_id == new_capsule_id:
                continue
                
            # Calculate similarity
            similarity = cosine_similarity([new_vector], [existing_data['vector']])[0][0]
            
            # Create connection if similarity is high enough
            if similarity > 0.3:  # Similarity threshold
                # Add bidirectional connections
                if existing_id not in self.connections[new_capsule_id]:
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
                
                print(f"🔗 Connected {new_capsule_id} ↔ {existing_id} (similarity: {similarity:.3f})")
    
    def get_all_capsules(self) -> List[Dict]:
        """Get all capsules"""
        return list(self.capsules.values())
    
    def get_capsule_connections(self, capsule_id: str) -> List[Dict]:
        """Get connections for a specific capsule"""
        return self.connections.get(capsule_id, [])
    
    def update_capsule_activation(self, capsule_id: str, activation: float, is_active: bool):
        """Update capsule activation state"""
        if capsule_id in self.capsules:
            self.capsules[capsule_id]['activation'] = activation
            self.capsules[capsule_id]['is_active'] = is_active
    
    def calculate_similarity(self, query_text: str, capsule_vector) -> float:
        """Calculate similarity between query and capsule"""
        query_vector = self.encoder.encode(query_text)
        similarity = cosine_similarity([query_vector], [capsule_vector])[0][0]
        return float(similarity)
    
    def update_connection_weight(self, capsule1_id: str, capsule2_id: str, new_weight: float):
        """Update connection weight between two capsules"""
        # Update weight in capsule1's connections
        if capsule1_id in self.connections:
            for conn in self.connections[capsule1_id]:
                if conn['connected_id'] == capsule2_id:
                    conn['weight'] = new_weight
        
        # Update weight in capsule2's connections
        if capsule2_id in self.connections:
            for conn in self.connections[capsule2_id]:
                if conn['connected_id'] == capsule1_id:
                    conn['weight'] = new_weight
    
    def save_database(self):
        """Save database to disk (simple pickle for now)"""
        import pickle
        
        data = {
            'capsules': self.capsules,
            'connections': self.connections
        }
        
        with open(os.path.join(self.db_path, 'capsules_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Database saved to {self.db_path}")
    
    def load_database(self):
        """Load database from disk"""
        import pickle
        
        data_file = os.path.join(self.db_path, 'capsules_data.pkl')
        
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.capsules = data['capsules']
            self.connections = data['connections']
            
            print(f"📂 Database loaded from {self.db_path}")
        else:
            print("📂 No existing database found, starting fresh")
    
    def close(self):
        """Close database connection and save data"""
        self.save_database()
        print("✅ Database closed and saved")
