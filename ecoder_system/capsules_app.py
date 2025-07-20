"""
Knowledge Capsules Application
Main application that creates and manages knowledge capsules
"""

from capsules_algorithm import KnowledgeCapsulesAlgorithm

class KnowledgeCapsulesApp:
    """
    Main application for Knowledge Capsules system
    Provides easy interface to create capsules and ask questions
    """
    
    def __init__(self):
        """Initialize the Knowledge Capsules system"""
        print("🌟 Initializing Knowledge Capsules System...")
        self.algorithm = KnowledgeCapsulesAlgorithm()
        self.setup_initial_capsules()
    
    def setup_initial_capsules(self):
        """Create initial knowledge capsules"""
        print("📦 Setting up initial knowledge capsules...")
        
        # Health & Wellness capsules
        health_capsules = [
            ("exercise", "Regular physical exercise strengthens muscles, improves heart health, and boosts mood through endorphin release", "health"),
            ("nutrition", "Balanced nutrition with fruits, vegetables, proteins, and whole grains provides essential vitamins and minerals for optimal body function", "health"),
            ("sleep", "Quality sleep of 7-9 hours allows body repair, memory consolidation, and immune system strengthening", "health"),
            ("hydration", "Adequate water intake maintains cellular function, regulates body temperature, and supports kidney health", "health"),
            ("stress_management", "Managing stress through meditation, deep breathing, and relaxation techniques improves mental and physical health", "health")
        ]
        
        # Technology capsules
        tech_capsules = [
            ("programming", "Programming involves writing instructions for computers using languages like Python, JavaScript, and Java", "technology"),
            ("ai_basics", "Artificial Intelligence uses algorithms and data to create systems that can perform tasks typically requiring human intelligence", "technology"),
            ("web_development", "Web development combines HTML for structure, CSS for styling, and JavaScript for interactivity to create websites", "technology")
        ]
        
        # Learning & Education capsules
        learning_capsules = [
            ("effective_learning", "Effective learning combines active practice, spaced repetition, and connecting new information to existing knowledge", "education"),
            ("time_management", "Good time management involves prioritizing tasks, setting realistic goals, and eliminating distractions", "education"),
            ("memory_techniques", "Memory techniques like mnemonics, visualization, and chunking help retain and recall information more effectively", "education")
        ]
        
        # Add all capsules to the system
        all_capsules = health_capsules + tech_capsules + learning_capsules
        
        for capsule_id, text, category in all_capsules:
            # Check if capsule already exists
            if capsule_id not in self.algorithm.db.capsules:
                self.algorithm.add_capsule(capsule_id, text, category)
        
        print(f"✅ Setup complete! Created {len(all_capsules)} knowledge capsules")
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get response from knowledge capsules"""
        print("\n" + "="*60)
        print(f"❓ Question: {question}")
        print("="*60)
        
        response, active_capsules = self.algorithm.process_query(question)
        
        print("\n🤖 Response:")
        print(f"{response}")
        print(f"\n🔥 Active Capsules: {active_capsules}")
        
        return response
    
    def give_feedback(self, feedback: str):
        """Provide feedback on the last response"""
        print(f"\n👤 Feedback: {feedback}")
        self.algorithm.learn_from_feedback(feedback)
    
    def add_custom_capsule(self, capsule_id: str, knowledge_text: str, category: str = "custom"):
        """Add a custom knowledge capsule"""
        self.algorithm.add_capsule(capsule_id, knowledge_text, category)
        print(f"✅ Added custom capsule: {capsule_id}")
    
    def show_system_status(self):
        """Show current system status"""
        self.algorithm.show_status()
        
        # Show connections for active capsules
        active_capsules = [c for c in self.algorithm.db.get_all_capsules() if c['is_active']]
        
        if active_capsules:
            print("\n🔗 Active capsule connections:")
            for capsule in active_capsules:
                connections = self.algorithm.db.get_capsule_connections(capsule['id'])
                if connections:
                    print(f"  {capsule['id']}:")
                    for conn in connections[:3]:  # Show top 3 connections
                        print(f"    → {conn['connected_id']} (strength: {conn['weight']:.3f})")
    
    def interactive_session(self):
        """Run interactive question-answer session"""
        print("\n🎯 Starting interactive session!")
        print("Type 'quit' to exit, 'status' to see system status")
        print("After each answer, you can say 'good' or 'bad' for feedback")
        
        while True:
            try:
                user_input = input("\n💬 Ask a question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'status':
                    self.show_system_status()
                    continue
                elif not user_input:
                    continue
                
                # Process the question
                self.ask_question(user_input)
                
                # Get feedback
                feedback = input("\n👍👎 Feedback (good/bad/skip): ").strip()
                if feedback and feedback.lower() != 'skip':
                    self.give_feedback(feedback)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        self.close()
    
    def close(self):
        """Close the application and save data"""
        print("\n💾 Saving and closing Knowledge Capsules system...")
        self.algorithm.close()
        print("✅ System closed successfully")

def main():
    """Main application entry point"""
    print("🚀 Knowledge Capsules - Intelligent Q&A System")
    print("=" * 50)
    
    # Create the application
    app = KnowledgeCapsulesApp()
    
    # Example usage
    print("\n📋 DEMO: Asking sample questions...")
    
    # Sample questions
    sample_questions = [
        "How can I stay healthy?",
        "What should I do to learn programming effectively?",
        "I'm feeling stressed, what can help?"
    ]
    
    for question in sample_questions:
        app.ask_question(question)
        
        # Simulate positive feedback for demo
        app.give_feedback("good")
        
        print("\n⏸️ Press Enter to continue...")
        input()
    
    # Show final system status
    app.show_system_status()
    
    # Ask if user wants interactive session
    print("\n🎮 Would you like to start an interactive session? (y/n)")
    if input().lower().startswith('y'):
        app.interactive_session()
    else:
        app.close()

if __name__ == "__main__":
    main()
