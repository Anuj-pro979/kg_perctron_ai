from ai_system_integration import ModularAISystem

# Initialize system
ai = ModularAISystem()

# Add knowledge
ai.add_knowledge("health_001", "Exercise improves cardiovascular health", "health")
ai.add_knowledge("tech_001", "Python is great for AI development", "technology")

# Process queries
response, metadata = ai.process_query("How can I improve my health?")
print(f"Response: {response}")

# Learn from feedback
ai.learn_from_feedback("good response")

# Close system
ai.close()
