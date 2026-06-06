"""
Main entry point for the MMAS Multi-Agent Query Router.

Usage:
    python "1Parallel MMAS Query Router/main.py"          # CLI mode
    python "1Parallel MMAS Query Router/main.py" --ui     # Gradio UI mode
"""

import sys
import os
import json
import datetime
import logging

# ── Windows UTF-8 fix for Arabic text ──
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Path resolution ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Load environment ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))
except ImportError:
    pass

# ── Configure MMAS logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_SCRIPT_DIR, "mmas.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Reduce noise from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger("mmas.main")

# ── Import graph ──
from graph import app
from langchain_core.messages import HumanMessage, AIMessage


def _empty_state(question: str, chat_history: list = None) -> dict:
    """Create a fresh initial state for the graph."""
    return {
        "question": question,
        "current_agent": "",
        "selected_agents": [],
        "retrieved_context": [],
        "reranker_score": 0.0,
        "sub_queries": [],
        "sub_query_agents": {},
        "final_answer": "",
        "messages": chat_history or [],
        "loop_step": 0,
        "colony_results": {},
        "colony_pheromones": {},
        "inspector_scores": {},
        "pheromone_log": [],
        "cache_hit": False,
        "cached_answer": "",
    }


def _log_evaluation(state: dict):
    """Append the final state details to a JSON evaluation log."""
    log_file = os.path.join(_SCRIPT_DIR, "evaluation_logs.json")

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
        "cache_hit": state.get("cache_hit", False),
        "inspector_scores": state.get("inspector_scores", {}),
        "pheromone_log": state.get("pheromone_log", []),
        "retrieved_documents_count": len(docs_meta),
        "retrieved_documents_metadata": docs_meta,
        "final_answer": state.get("final_answer", ""),
    }

    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            pass

    logs.append(eval_record)

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write evaluation log: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Mode
# ═══════════════════════════════════════════════════════════════════════════════

def print_welcome():
    print("=" * 60)
    print("       Welcome to Yaqeen — MMAS Query Router")
    print("  Ant Colony Optimized | Multi-Agent | Semantic Cache")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)


def main():
    print_welcome()
    chat_history = []

    while True:
        try:
            user_input = input("\nUser: ")

            if user_input.strip().lower() in ["quit", "exit"]:
                print("\nShutting down Yaqeen. Goodbye!")
                break

            if not user_input.strip():
                continue

            initial_state = _empty_state(user_input, chat_history)

            print("\nThinking...")
            final_state = app.invoke(initial_state)

            answer = final_state.get(
                "final_answer",
                "I encountered an error generating a response.",
            )

            # Show cache hit indicator
            if final_state.get("cache_hit"):
                print("\n⚡ Answer from cache:")

            print(f"\nYaqeen: {answer}")

            # Log evaluation
            _log_evaluation(final_state)

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Exiting.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\n[Error]: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio UI Mode
# ═══════════════════════════════════════════════════════════════════════════════

def _run_gradio():
    try:
        import gradio as gr
    except ImportError:
        print("[Error] Gradio not installed. Run: pip install gradio")
        return

    def query(user_input, chat_history, log_text):
        """Process a user query and return updated chat and log."""
        if not user_input.strip():
            return chat_history, log_text

        init_state = _empty_state(user_input)
        final_state = app.invoke(init_state)
        answer = final_state.get("final_answer", "[Error] No answer produced.")

        _log_evaluation(final_state)

        # Build pheromone log display
        pheromone_log = final_state.get("pheromone_log", [])
        if pheromone_log:
            log_lines = [f"--- Query: {user_input[:50]}... ---"]
            for entry in pheromone_log:
                log_lines.append(
                    f"Colony: {entry.get('colony_id', '?')} | "
                    f"Best: {entry.get('best_worker', '?')}({entry.get('best_score', 0):.3f}) | "
                    f"τ_min={entry.get('tau_min', 0):.4f}, τ_max={entry.get('tau_max', 0):.4f}"
                )
            new_log = "\n".join(log_lines)
            log_text = f"{new_log}\n\n{log_text}" if log_text else new_log

        cache_indicator = "⚡ [cached] " if final_state.get("cache_hit") else ""
        return chat_history + [(user_input, f"{cache_indicator}{answer}")], log_text

    with gr.Blocks(title="Yaqeen — MMAS Query Router") as demo:
        gr.Markdown("# 🐜 Yaqeen — Ant Colony Optimized Islamic Knowledge Assistant")
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Chat")
                txt = gr.Textbox(
                    label="Ask Yaqeen",
                    placeholder="Type your question...",
                    lines=1,
                )
                btn = gr.Button("Send")
            with gr.Column(scale=1):
                log = gr.Textbox(
                    label="MMAS Pheromone Log",
                    lines=20,
                    interactive=False,
                )

        txt.submit(query, inputs=[txt, chat, log], outputs=[chat, log])
        btn.click(query, inputs=[txt, chat, log], outputs=[chat, log])
        demo.launch()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        _run_gradio()
    else:
        main()