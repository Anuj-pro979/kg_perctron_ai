"""
Transformer-based Decoder for Knowledge Capsules
Uses a different transformer model for decoding responses
"""

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import torch

class TransformerDecoder:
    """
    Advanced decoder using HuggingFace transformers
    Separate from encoder for true modular architecture
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """Initialize decoder with different transformer"""
        print("🤖 Loading transformer decoder...")
        try:
            # Use text generation pipeline
            self.generator = pipeline(
                "text-generation", 
                model="gpt2",  # Lightweight but effective
                tokenizer="gpt2",
                device=0 if torch.cuda.is_available() else -1,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            print("✅ GPT-2 decoder loaded")
        except:  # noqa: E722
            # Fallback to basic generation
            print("⚠️ Using fallback decoder")
            self.generator = None
        
        # Response enhancement templates
        self.templates = {
            'health': "For health and wellness: {content}",
            'technology': "Regarding technology: {content}", 
            'education': "For learning: {content}",
            'general': "Based on knowledge: {content}"
        }
    
    def decode_response(self, query: str, active_capsules: List[Dict], 
                       encoder_vector: np.ndarray) -> str:
        """
        Main decoding function using transformer
        """
        if not active_capsules:
            return "I don't have sufficient knowledge to answer that question."
        
        # Prepare context from capsules
        context = self._prepare_context(active_capsules)
        
        # Generate response using transformer
        if self.generator:
            response = self._generate_with_transformer(query, context)
        else:
            response = self._generate_fallback(query, context)
        
        # Enhance with capsule confidence
        enhanced_response = self._enhance_with_confidence(response, active_capsules)
        
        return enhanced_response
    
    def _prepare_context(self, active_capsules: List[Dict]) -> str:
        """Prepare context from active capsules"""
        # Sort by activation
        sorted_capsules = sorted(active_capsules, key=lambda x: x['activation'], reverse=True)
        
        # Take top 3 capsules
        top_capsules = sorted_capsules[:3]
        
        context_parts = []
        for capsule in top_capsules:
            context_parts.append(capsule['text'])
        
        return ". ".join(context_parts)
    
    def _generate_with_transformer(self, query: str, context: str) -> str:
        """Generate response using transformer model"""
        # Create prompt
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        
        try:
            # Generate response
            generated = self.generator(
                prompt, 
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            # Extract the answer part
            full_response = generated[0]['generated_text']
            answer = full_response.replace(prompt, "").strip()
            
            # Clean up the response
            if answer:
                # Remove any incomplete sentences
                sentences = answer.split('.')
                complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                if complete_sentences:
                    return '. '.join(complete_sentences[:2]) + '.'
            
            # Fallback if transformer response is poor
            return self._generate_fallback(query, context)
        
        except Exception as e:
            print(f"⚠️ Transformer generation failed: {e}")
            return self._generate_fallback(query, context)
    
    def _generate_fallback(self, query: str, context: str) -> str:
        """Fallback response generation"""
        # Identify query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'what should']):
            return f"To address your question: {context}. This approach should help you achieve your goal."
        elif 'why' in query_lower:
            return f"The explanation is: {context}. This provides the reasoning behind the concept."
        elif any(word in query_lower for word in ['what', 'which']):
            return f"Based on available knowledge: {context}. This information directly addresses your question."
        else:
            return f"Here's what I know: {context}. This should provide helpful insights for your situation."
    
    def _enhance_with_confidence(self, response: str, active_capsules: List[Dict]) -> str:
        """Enhance response with confidence and metadata"""
        # Calculate average confidence
        if active_capsules:
            avg_confidence = sum(c['activation'] for c in active_capsules) / len(active_capsules)
            confidence_level = "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.5 else "Moderate"
            
            # Get unique categories
            categories = list(set(c.get('category', 'general') for c in active_capsules))
            
            # Add metadata
            enhanced = f"{response}\n\n[Confidence: {confidence_level} | Sources: {len(active_capsules)} capsules | Categories: {', '.join(categories)}]"
            return enhanced
        
        return response
    
    def generate_contextual_response(self, query: str, capsule_texts: List[str], 
                                   activation_scores: List[float]) -> str:
        """Alternative generation method with explicit inputs"""
        if not capsule_texts:
            return "No relevant knowledge available."
        
        # Weight texts by activation scores
        weighted_context = []
        for text, score in zip(capsule_texts, activation_scores):
            if score > 0.4:  # Only include relevant capsules
                weighted_context.append(f"{text} (relevance: {score:.2f})")
        
        context = " | ".join(weighted_context)
        
        if self.generator:
            return self._generate_with_transformer(query, context)
        else:
            return self._generate_fallback(query, context)

class MinimalDecoder:
    """
    Ultra-minimal decoder for rapid prototyping
    No external dependencies beyond numpy
    """
    
    def __init__(self):
        """Initialize minimal decoder"""
        print("⚡ Minimal decoder ready")
        self.response_patterns = {
            'how': "Here's how to {topic}: {content}. Follow these principles for best results.",
            'what': "Regarding {topic}: {content}. This covers the key aspects.",
            'why': "The reason for {topic} is: {content}. This explains the underlying mechanism.",
            'when': "For timing of {topic}: {content}. Consider these factors.",
            'default': "About {topic}: {content}. This should help with your question."
        }
    
    def decode_response(self, query: str, active_capsules: List[Dict], 
                       encoder_vector: np.ndarray = None) -> str:
        """Minimal but effective decoding"""
        if not active_capsules:
            return "I need more specific information to answer that question."
        
        # Get question word
        query_lower = query.lower()
        question_type = 'default'
        for qtype in ['how', 'what', 'why', 'when']:
            if qtype in query_lower:
                question_type = qtype
                break
        
        # Extract topic from query
        topic = self._extract_topic(query)
        
        # Get best capsule content
        best_capsule = max(active_capsules, key=lambda x: x['activation'])
        content = best_capsule['text']
        
        # Generate response
        pattern = self.response_patterns[question_type]
        response = pattern.format(topic=topic, content=content)
        
        # Add confidence
        confidence = best_capsule['activation']
        conf_text = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
        
        return f"{response}\n\n(Confidence: {conf_text} | Knowledge source: {best_capsule['id']})"
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query"""
        # Simple topic extraction
        important_words = []
        skip_words = {'how', 'what', 'why', 'when', 'can', 'should', 'is', 'are', 'do', 'does', 'i', 'my', 'me'}
        
        for word in query.lower().split():
            cleaned_word = word.strip('.,!?')
            if cleaned_word not in skip_words and len(cleaned_word) > 2:
                important_words.append(cleaned_word)
        
        return ' '.join(important_words[:3]) if important_words else "this topic"
