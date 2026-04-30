from dotenv import load_dotenv
load_dotenv()

from graph import app
from state import AgentState

def print_welcome():
    print("="*60)
    print("          Welcome to VectorMind CLI")
    print("  Type 'quit' or 'exit' to end the session.")
    print("="*60)

def main():
    print_welcome()
    
    # Initialize rolling message history for the session
    chat_history = []
    
    while True:
        try:
            # 1. Get User Input
            user_input = input("\nUser: ")
            
            # 2. Check for exit commands
            if user_input.strip().lower() in ['quit', 'exit']:
                print("\nShutting down VectorMind. Goodbye!")
                break
                
            if not user_input.strip():
                continue

            # 3. Initialize State Payload
            # We pass the current chat_history and reset temporary variables
            initial_state = {
                "question": user_input,
                "current_agent": "",
                "retrieved_context": [],
                "reranker_score": 0.0,
                "sub_queries": [],
                "final_answer": "",
                "messages": chat_history,
                "loop_step": 0
            }

            print("\nThinking...")
            
            # 4. Execute the Graph
            final_state = app.invoke(initial_state)
            
            # 5. Extract and Display Output
            answer = final_state.get("final_answer", "I encountered an error generating a response.")
            print(f"\nVectorMind: {answer}")
            
            # 6. Update Chat History
            # If you want the bot to remember context in future iterations,
            # you would append the interaction here. (Assuming your agents
            # are configured to read the 'messages' list).
            # chat_history.append(HumanMessage(content=user_input))
            # chat_history.append(AIMessage(content=answer))

        except KeyboardInterrupt:
            # Handles CTRL+C gracefully
            print("\n\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error]: An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()