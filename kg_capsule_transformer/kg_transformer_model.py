import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
import logging

# Import our KGCapsule (assuming it's in the same directory)
from kg_capsule_module import KGCapsuleTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KGEnhancedTransformer(nn.Module):
    """
    Knowledge-Enhanced Transformer with Sandwich Architecture
    
    Architecture: Frozen Encoder → KGCapsule → Frozen Decoder
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        decoder_name: str = "distilbert-base-uncased", 
        max_knowledge_nodes: int = 500,
        top_k_retrieval: int = 5,
        knowledge_learning_rate: float = 1e-4
    ):
        super().__init__()
        
        # Load and freeze encoder (HuggingFace transformer)
        logger.info(f"Loading frozen encoder: {encoder_name}")
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info(f"Encoder frozen: {sum(p.numel() for p in self.encoder.parameters())} parameters")
        # Load and freeze decoder
        logger.info(f"Loading frozen decoder: {decoder_name}")
        self.decoder = AutoModel.from_pretrained(decoder_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        # Freeze decoder parameters  
        for param in self.decoder.parameters():
            param.requires_grad = False
        logger.info(f"Decoder frozen: {sum(p.numel() for p in self.decoder.parameters())} parameters")
        # Get model dimensions
        self.encoder_dim = self.encoder.config.hidden_size
        decoder_config = self.decoder.config
        self.decoder_dim = decoder_config.hidden_size
        logger.info(f"Encoder dim: {self.encoder_dim}, Decoder dim: {self.decoder_dim}")
        
        # Dimension alignment layer (if needed)
        if self.encoder_dim != self.decoder_dim:
            self.dimension_adapter = nn.Linear(self.encoder_dim, self.decoder_dim)
            logger.info(f"Added dimension adapter: {self.encoder_dim} -> {self.decoder_dim}")
        else:
            self.dimension_adapter = nn.Identity()
        
        # Knowledge Graph Capsule (trainable!)
        self.kg_capsule = KGCapsuleTransformer(
            hidden_dim=self.decoder_dim,
            max_knowledge_nodes=max_knowledge_nodes,
            top_k_retrieval=top_k_retrieval,
            confidence_threshold=0.6
        )
        
        # Task-specific head
        self.task_head = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.decoder_dim // 2, self.decoder_dim)
        )
        
        # Store learning rate for optimizer setup
        self.knowledge_lr = knowledge_learning_rate
        
        logger.info("KG-Enhanced Transformer initialized!")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_knowledge: bool = True,
        return_kg_stats: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the sandwich architecture
        
        Args:
            input_ids: [batch_size, seq_len] 
            attention_mask: [batch_size, seq_len]
            update_knowledge: Whether to update KG during training
            return_kg_stats: Whether to return knowledge statistics
            
        Returns:
            Dictionary with outputs and optional statistics
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. FROZEN ENCODER: Extract representations
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            encoder_hidden = encoder_outputs.last_hidden_state  # [batch, seq_len, encoder_dim]
        
        # 2. DIMENSION ALIGNMENT: Match encoder to decoder dimensions
        aligned_hidden = self.dimension_adapter(encoder_hidden)  # [batch, seq_len, decoder_dim]
        
        # 3. KNOWLEDGE ENHANCEMENT: Our trainable KG Capsule!
        enhanced_hidden, kg_stats = self.kg_capsule(
            sequence=aligned_hidden,
            attention_mask=attention_mask,
            update_knowledge=update_knowledge
        )
        
        # 4. FROZEN DECODER: Generate contextual representations  
        with torch.no_grad():
            decoder_outputs = self.decoder(
                inputs_embeds=enhanced_hidden,
                attention_mask=attention_mask,
                return_dict=True
            )
            decoder_hidden = decoder_outputs.last_hidden_state  # [batch, seq_len, decoder_dim]
        
        # 5. TASK HEAD: Final processing
        task_output = self.task_head(decoder_hidden)  # [batch, seq_len, decoder_dim]
        
        # Prepare return dictionary
        outputs = {
            'last_hidden_state': task_output,
            'enhanced_hidden': enhanced_hidden,
            'pooled_output': task_output.mean(dim=1)  # Simple pooling for classification tasks
        }
        
        if return_kg_stats:
            outputs['kg_stats'] = kg_stats
            outputs['knowledge_summary'] = self.kg_capsule.get_knowledge_stats()
        
        return outputs
    
    def get_optimizer(self) -> optim.Optimizer:
        """Get optimizer for trainable parameters only"""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
        return optim.AdamW(trainable_params, lr=self.knowledge_lr, weight_decay=0.01)
    
    def save_model(self, save_path: str):
        """Save the model state"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'kg_stats': self.kg_capsule.get_knowledge_stats(),
            'config': {
                'encoder_dim': self.encoder_dim,
                'decoder_dim': self.decoder_dim,
                'knowledge_nodes': self.kg_capsule.current_nodes
            }
        }, os.path.join(save_path, 'kg_model.pt'))
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the model state"""
        checkpoint = torch.load(os.path.join(load_path, 'kg_model.pt'))
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {load_path}")
        logger.info(f"Knowledge stats: {checkpoint['kg_stats']}")


class SimpleTextDataset(Dataset):
    """
    Simple dataset for text processing tasks
    Works with any text data - can be used for various tasks
    """
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer, 
        max_length: int = 128,
        task_type: str = "reconstruction"
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        logger.info(f"Dataset created: {len(texts)} samples, task: {task_type}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),  # Self-supervised
            'text': text
        }


def create_sample_dataset(size: int = 1000) -> List[str]:
    """Create a diverse sample dataset for testing"""
    
    # Knowledge-rich sample texts covering various domains
    sample_texts = [
        # Science & Technology
        "Artificial intelligence is transforming healthcare through machine learning algorithms that can diagnose diseases.",
        "Quantum computing uses quantum mechanical phenomena to process information exponentially faster than classical computers.",
        "Climate change is caused by increased greenhouse gas emissions from human industrial activities.",
        "CRISPR gene editing technology allows precise modification of DNA sequences in living organisms.",
        "Solar panels convert sunlight into electricity through photovoltaic cells made of silicon semiconductors.",
        
        # History & Geography  
        "The Roman Empire reached its peak under Emperor Trajan, spanning three continents and lasting over 1000 years.",
        "The Amazon rainforest produces 20% of the world's oxygen and contains incredible biodiversity.",
        "World War II ended in 1945 with the surrender of Germany and Japan to Allied forces.",
        "Mount Everest is the highest mountain on Earth, located in the Himalayas between Nepal and Tibet.",
        "The Renaissance period brought significant advances in art, science, and philosophy in 14th-17th century Europe.",
        
        # Business & Economics
        "Supply and demand determine market prices in free market economic systems.",
        "Cryptocurrency uses blockchain technology to create decentralized digital currencies.",
        "Startups require venture capital funding to scale their innovative business models.",
        "Global trade connects countries through imports and exports of goods and services.",
        "Inflation occurs when the general price level of goods and services increases over time.",
        
        # Culture & Society
        "Language evolution shows how human communication develops through social interaction and cultural change.",
        "Democracy requires active citizen participation in voting and civic engagement.",
        "Education systems must adapt to prepare students for rapidly changing job markets.",
        "Social media platforms have revolutionized how people connect and share information globally.",
        "Cultural diversity enriches societies through different perspectives, traditions, and innovations."
    ]
    
    # Extend the dataset by creating variations
    extended_texts = []
    for _ in range(size):
        base_text = np.random.choice(sample_texts)
        
        # Add some variations to create diverse training data
        variations = [
            base_text,
            f"According to recent studies, {base_text.lower()}",
            f"It is important to note that {base_text.lower()}",
            f"Research indicates that {base_text.lower()}",
            f"Experts believe that {base_text.lower()}"
        ]
        
        extended_texts.append(np.random.choice(variations))
    
    return extended_texts


def train_kg_model(
    model: KGEnhancedTransformer,
    dataloader: DataLoader,
    num_epochs: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, List[float]]:
    """
    Train the KG-Enhanced Transformer
    
    Args:
        model: The KG-Enhanced Transformer model
        dataloader: Training data loader
        num_epochs: Number of training epochs
        device: Training device
        
    Returns:
        Training history with losses and metrics
    """
    logger.info(f"Training on device: {device}")
    model = model.to(device)
    model.train()
    
    # Setup optimizer
    optimizer = model.get_optimizer()
    criterion = nn.MSELoss()  # Simple reconstruction loss
    
    # Training history
    history = {
        'losses': [],
        'kg_nodes': [],
        'avg_confidence': [],
        'knowledge_usage': []
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_kg_stats = []
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                update_knowledge=True,
                return_kg_stats=True
            )
            
            # Simple reconstruction loss (self-supervised)
            # In practice, you'd use task-specific loss
            hidden_states = outputs['last_hidden_state']
            target_embeddings = model.dimension_adapter(
                model.encoder(input_ids, attention_mask).last_hidden_state.detach()
            )
            
            loss = criterion(hidden_states, target_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_kg_stats.append(outputs['kg_stats'])
            
            # Update progress
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'KG_Nodes': outputs['knowledge_summary']['total_nodes'],
                'Confidence': f"{outputs['kg_stats']['avg_confidence']:.3f}"
            })
            
            # Process knowledge updates every few batches
            if batch_idx % 10 == 0:
                model.kg_capsule._process_knowledge_updates()
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_nodes = np.mean([stats['active_nodes'] for stats in epoch_kg_stats])
        avg_confidence = np.mean([stats['avg_confidence'] for stats in epoch_kg_stats])
        
        history['losses'].append(avg_loss)
        history['kg_nodes'].append(avg_nodes)
        history['avg_confidence'].append(avg_confidence)
        
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Knowledge Nodes: {avg_nodes:.1f}")
        logger.info(f"  Avg Confidence: {avg_confidence:.3f}")
        
        # Prune unused knowledge periodically
        if epoch % 2 == 0:
            pruned = model.kg_capsule.prune_unused_knowledge()
            logger.info(f"  Pruned {pruned} unused knowledge nodes")
    
    return history


def evaluate_model(
    model: KGEnhancedTransformer,
    test_texts: List[str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Evaluate the trained model"""
    logger.info("Evaluating model...")
    model = model.to(device)
    model.eval()
    
    tokenizer = model.decoder_tokenizer
    
    results = []
    with torch.no_grad():
        for text in test_texts[:5]:  # Test on first 5 samples
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                update_knowledge=False,
                return_kg_stats=True
            )
            
            results.append({
                'text': text,
                'kg_stats': outputs['kg_stats'],
                'knowledge_retrieved': outputs['kg_stats']['knowledge_retrieved'],
                'confidence': outputs['kg_stats']['avg_confidence']
            })
    
    # Print results
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nSample {i}:")
        print(f"Text: {result['text'][:100]}...")
        print(f"Knowledge Retrieved: {result['knowledge_retrieved']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"KG Stats: {result['kg_stats']}")
    
    # Final model statistics
    final_stats = model.kg_capsule.get_knowledge_stats()
    print("\n" + "="*50)
    print("FINAL KNOWLEDGE GRAPH STATISTICS")
    print("="*50)
    for key, value in final_stats.items():
        print(f"{key}: {value}")


def main():
    """Main training and evaluation pipeline"""
    logger.info("🚀 Starting KG-Enhanced Transformer Training")
    
    # Configuration
    config = {
        'batch_size': 8,
        'max_length': 128,
        'num_epochs': 3,
        'dataset_size': 200,  # Small for testing
        'learning_rate': 1e-4,
        'max_knowledge_nodes': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Configuration: {config}")
    
    # 1. Create dataset
    logger.info("📚 Creating dataset...")
    texts = create_sample_dataset(size=config['dataset_size'])
    
    # 2. Initialize model
    logger.info("🏗️ Initializing KG-Enhanced Transformer...")
    model = KGEnhancedTransformer(
        max_knowledge_nodes=config['max_knowledge_nodes'],
        knowledge_learning_rate=config['learning_rate']
    )
    
    # 3. Create dataloader
    tokenizer = model.decoder_tokenizer
    dataset = SimpleTextDataset(texts, tokenizer, max_length=config['max_length'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # 4. Train model
    logger.info("🎯 Starting training...")
    history = train_kg_model(
        model=model,
        dataloader=dataloader,
        num_epochs=config['num_epochs'],
        device=config['device']
    )
    
    # 5. Save model
    logger.info("💾 Saving model...")
    save_path = "./kg_model_checkpoint"
    model.save_model(save_path)
    
    # 6. Evaluate model
    logger.info("📊 Evaluating model...")
    test_texts = texts[-10:]  # Use last 10 samples for testing
    evaluate_model(model, test_texts, config['device'])
    
    # 7. Print training summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Final Loss: {history['losses'][-1]:.4f}")
    print(f"Knowledge Nodes Created: {history['kg_nodes'][-1]:.0f}")
    print(f"Average Confidence: {history['avg_confidence'][-1]:.3f}")
    print(f"Model saved to: {save_path}")
    
    logger.info("✅ Training completed successfully!")
    
    return model, history


if __name__ == "__main__":
    # Run the complete pipeline
    trained_model, training_history = main()