# test_kg_model.py - Quick test script
import torch

def quick_test():
    """Quick test to ensure everything works"""
    print("🔧 Testing KG-Enhanced Transformer...")
    
    try:
        # Test imports
        from kg_transformer_model import KGEnhancedTransformer, create_sample_dataset
        print("✅ Imports successful")
        
        # Test model initialization
        model = KGEnhancedTransformer(
            max_knowledge_nodes=50,
            knowledge_learning_rate=1e-4
        )
        print("✅ Model initialization successful")
        
        # Test forward pass
        tokenizer = model.decoder_tokenizer
        test_text = "Artificial intelligence is transforming the world through machine learning."
        
        encoding = tokenizer(
            test_text,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                update_knowledge=False,
                return_kg_stats=True
            )
        
        print("✅ Forward pass successful")
        print(f"Output shape: {outputs['last_hidden_state'].shape}")
        print(f"KG Stats: {outputs['kg_stats']}")
        
        # Test dataset creation
        sample_texts = create_sample_dataset(size=10)
        print(f"✅ Dataset creation successful: {len(sample_texts)} samples")
        
        print("\n🎉 All tests passed! Ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 Run 'python kg_transformer_model.py' to start training!")
    else:
        print("\n⚠️ Fix the issues above before training.")