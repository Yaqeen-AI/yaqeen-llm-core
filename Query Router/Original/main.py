import sys
import os
import json
import datetime

# Fix Windows console encoding for Arabic text output
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # not all stdout implementations support reconfigure

# Ensure Query Router directory is on sys.path so relative imports
# (graph, state, models.*, workers.*) resolve when running from project root:
#   python "Query Router/main.py"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Also add project root so `core.cache` is importable
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

# Load .env from the Query Router directory (contains HF_TOKEN etc.)
# Load .env if dotenv is available
if load_dotenv:
    load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))


from graph import app
from state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

# ── Cache integration (Tier 1: Redis exact-match, Tier 2: Qdrant semantic) ──
_cache = None

def _get_cache():
    """Lazy-init the TwoTierCache — graceful fallback if Redis/Qdrant unavailable."""
    global _cache
    if _cache is None:
        try:
            from core.cache import TwoTierCache
            _cache = TwoTierCache()
        except Exception as e:
            print(f"[Cache] Disabled — {e}")
            _cache = False  # sentinel: don't retry
    return _cache if _cache else None


def _log_evaluation(state: dict):
    """Append the final state details to a JSON evaluation log."""
    log_file = os.path.join(_SCRIPT_DIR, "evaluation_logs.json")
    
    # Extract only metadata to avoid massive text bloat
    docs_meta = []
    for doc in state.get("retrieved_context", []):
        meta = dict(doc.metadata) if hasattr(doc, "metadata") else {}
        docs_meta.append(meta)
        
    eval_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "original_question": state.get("question", ""),
        "sub_queries": state.get("sub_queries", []),
        "sub_query_agents": state.get("sub_query_agents", {}),
        "selected_agents": state.get("selected_agents", []),
        "retrieved_documents_count": len(docs_meta),
        "retrieved_documents_metadata": docs_meta,
        "final_answer": state.get("final_answer", "")
    }
    
    # Read existing logs if file exists
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            pass
            
    logs.append(eval_record)
    
    # Write back
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Evaluation Logger] Failed to write log: {e}")


def print_welcome():
    print("="*60)
    print("          Welcome to Yaqeen CLI")
    print("  Multi-Query Router | Keyword-Based | Cached")
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
                print("\nShutting down Yaqeen. Goodbye!")
                break
                
            if not user_input.strip():
                continue

            # 3. Cache check — return immediately on hit
            cache = _get_cache()
            if cache:
                cached_answer = cache.get(user_input)
                if cached_answer is not None:
                    print("\n⚡ Answer from cache:")
                    print(f"\nYaqeen: {cached_answer}")
                    continue

            # 4. Initialize State Payload
            # We pass the current chat_history and reset temporary variables
            initial_state = {
                "question": user_input,
                "current_agent": "",
                "selected_agents": [],
                "retrieved_context": [],
                "reranker_score": 0.0,
                "sub_queries": [],
                "sub_query_agents": {},
                "final_answer": "",
                "messages": chat_history,
                "loop_step": 0
            }

            print("\nThinking...")
            
            # 5. Execute the Graph
            final_state = app.invoke(initial_state)
            
            # 6. Extract and Display Output
            answer = final_state.get("final_answer", "I encountered an error generating a response.")
            print(f"\nYaqeen: {answer}")

            # 6.5 Log Evaluation
            _log_evaluation(final_state)

            # 7. Cache store — save answer for future lookups
            if cache and answer:
                try:
                    cache.set(user_input, answer)
                except Exception:
                    pass  # don't break the flow if cache write fails
            
            # 8. Update Chat History
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        except KeyboardInterrupt:
            # Handles CTRL+C gracefully
            print("\n\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error]: An unexpected error occurred: {str(e)}")

def _run_gradio():
    try:
        import gradio as gr
    except ImportError:
        print("[Error] Gradio is not installed. Install it with `pip install gradio`.")
        return

    def query(user_input, chat_history, log_text):
        """Process a user query and return updated chat and log.
        The function now returns only the final answer without extra debug logs.
        """
        if not user_input.strip():
            return chat_history, log_text
        # Retrieve from cache if possible
        cache = _get_cache()
        if cache:
            cached = cache.get(user_input)
            if cached is not None:
                answer = cached
                # No internal log output for cache hit
                # Update chat history and keep existing log unchanged
                return chat_history + [(user_input, answer)], log_text
        # No cache hit; invoke graph to get answer
        init_state = {
            "question": user_input,
            "current_agent": "",
            "selected_agents": [],
            "retrieved_context": [],
            "reranker_score": 0.0,
            "sub_queries": [],
            "sub_query_agents": {},
            "final_answer": "",
            "messages": [],
            "loop_step": 0,
        }
        final_state = app.invoke(init_state)
        answer = final_state.get("final_answer", "[Error] No answer produced.")
        _log_evaluation(final_state)
        if cache and answer:
            try:
                cache.set(user_input, answer)
            except Exception:
                pass
        # Append answer to chat and return unchanged log
        return chat_history + [(user_input, answer)], log_text

    with gr.Blocks(title="Yaqeen – Islamic Knowledge Assistant") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Chat")
                txt = gr.Textbox(label="Ask Yaqeen", placeholder="Type your question...", lines=1)
                btn = gr.Button("Send")
            with gr.Column(scale=1):
                log = gr.Textbox(label="Processing Log", lines=20, interactive=False)
        # Wire events after all components exist
        txt.submit(query, inputs=[txt, chat, log], outputs=[chat, log])
        btn.click(query, inputs=[txt, chat, log], outputs=[chat, log])
        demo.launch()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        _run_gradio()
    else:
        main()