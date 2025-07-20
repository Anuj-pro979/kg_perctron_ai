"""
Modular Encoder - Extracted from embedded_neo4j_setup.py
Handles all encoding operations separately for better modularity
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class ModularEncoder:
    """
    Separated encoder module using SentenceTransformer
    This replaces the encoder logic from EmbeddedNeo4jConnection
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the encoder"""
        print("🧠 Loading modular encoder...")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"✅ Encoder ready - Dimension: {self.embedding_dim}")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode single text to vector"""
        return self.encoder.encode(text)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts"""
        return self.encoder.encode(texts)
    
    def calculate_similarity(self, text1: str, vector2: np.ndarray) -> float:
        """Calculate similarity between text and vector"""
        vector1 = self.encode(text1)
        similarity = cosine_similarity([vector1], [vector2])[0][0]
        return float(similarity)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        vectors = self.encode_batch([text1, text2])
        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        return float(similarity)
