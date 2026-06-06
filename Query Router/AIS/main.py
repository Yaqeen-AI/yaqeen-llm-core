"""
Main entry point for the AIS (Artificial Immune System) Query Router.

Usage:
    python "Query Router/AIS/main.py"          # CLI mode
    python "Query Router/AIS/main.py" --ui     # Gradio UI mode
"""

import sys
import os
import json
import datetime
import logging
import asyncio

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

# ── Configure AIS logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_SCRIPT_DIR, "ais.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger("ais.main")

# ── Import graph ──
from graph import app
from state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


def _empty_state(question: str) -> dict:
    """Create a fresh initial state for the AIS graph."""
    return {
        "question": question,
        "sub_queries": [],
        "initial_context": [],
        "clones": [],
        "matured_sub_queries": [],
        "secondary_context": [],
        "suppressed_context": [],
        "final_answer": "",
        "cache_hit": False,
        "cached_answer": "",
        "loop_step": 0,
    }


def _log_evaluation(state: dict):
    """Append the final state details to a JSON evaluation log."""
    log_file = os.path.join(_SCRIPT_DIR, "evaluation_logs.json")

    docs_meta = []
    for doc in state.get("suppressed_context", []):
        meta = dict(doc.metadata) if hasattr(doc, "metadata") else {}
        docs_meta.append(meta)

    eval_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "antigen_query": state.get("question", ""),
        "cache_hit": state.get("cache_hit", False),
        "sub_queries": state.get("sub_queries", []),
        "clones_count": len(state.get("clones", [])),
        "matured_sub_queries": state.get("matured_sub_queries", []),
        "secondary_context_count": len(state.get("secondary_context", [])),
        "suppressed_context_count": len(docs_meta),
        "suppressed_context_metadata": docs_meta,
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
    print("       Welcome to Yaqeen — AIS Query Router")
    print("   Artificial Immune System | Async | Semantic Cache")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)


def main():
    print_welcome()

    while True:
        try:
            user_input = input("\nUser: ")

            if user_input.strip().lower() in ["quit", "exit"]:
                print("\nShutting down Yaqeen. Goodbye!")
                break

            if not user_input.strip():
                continue

            initial_state = _empty_state(user_input)

            print("\nThinking...")
            # Run the async graph using asyncio.run
            final_state = asyncio.run(app.ainvoke(initial_state))

            answer = final_state.get(
                "final_answer",
                "I encountered an error generating a response.",
            )

            if final_state.get("cache_hit"):
                print("\n⚡ [Memory Cells HIT] Answer from cache:")
            else:
                print(f"\n⚡ [Primary Immune Response Activated]")

            print(f"\nYaqeen: {answer}")

            # Log evaluation
            _log_evaluation(final_state)

        except KeyboardInterrupt:
            # Handles CTRL+C gracefully
            print("\n\nSession interrupted. Exiting.")
            break
        except EOFError:
            print("\nStdin closed. Exiting.")
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
        """Process a user query and return updated chat and immune log."""
        if not user_input.strip():
            return chat_history, log_text

        init_state = _empty_state(user_input)
        final_state = asyncio.run(app.ainvoke(init_state))
        answer = final_state.get("final_answer", "[Error] No answer produced.")

        _log_evaluation(final_state)

        # Build immune response log display
        log_lines = [f"--- Antigen: {user_input[:50]}... ---"]
        
        if final_state.get("cache_hit"):
            log_lines.append("🛡️ Memory Cells HIT: Returned cached answer.")
        else:
            log_lines.append("🛡️ Memory Cells MISS: Primary Immune Response activated.")
            
            sub_q = final_state.get("sub_queries", [])
            log_lines.append(f"🧬 Innate Response: Decomposed into {len(sub_q)} sub-queries:")
            for sq in sub_q:
                log_lines.append(f"   - '{sq}'")
                
            initial_count = len(final_state.get("initial_context", []))
            log_lines.append(f"🧬 Antibody Generation: Gathered {initial_count} initial chunks.")
            
            clones = final_state.get("clones", [])
            log_lines.append(f"🧬 Clonal Selection: Selected top {len(clones)} highest-affinity clones.")
            
            matured_q = final_state.get("matured_sub_queries", [])
            log_lines.append(f"🧬 Maturation: Expanded sub-queries to retrieve specific context.")
            
            sec_count = len(final_state.get("secondary_context", []))
            log_lines.append(f"🧬 Secondary Retrieval: Fetched {sec_count} matured antibodies.")
            
            suppressed = len(final_state.get("suppressed_context", []))
            log_lines.append(f"🧬 Suppression: Deduplicated to {suppressed} diverse context antibodies.")
            log_lines.append("🧬 Memory Cell Formation: Logged to Semantic Cache.")
            
        new_log = "\n".join(log_lines)
        log_text = f"{new_log}\n\n{log_text}" if log_text else new_log

        cache_indicator = "⚡ [Memory Cell Hit] " if final_state.get("cache_hit") else ""
        return chat_history + [(user_input, f"{cache_indicator}{answer}")], log_text

    with gr.Blocks(title="Yaqeen — AIS Query Router") as demo:
        gr.Markdown("# 🛡️ Yaqeen — Artificial Immune System (AIS) Optimized Islamic RAG")
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Chat")
                txt = gr.Textbox(
                    label="Ask Yaqeen (Antigen)",
                    placeholder="Type your question...",
                    lines=1,
                )
                btn = gr.Button("Send")
            with gr.Column(scale=1):
                log = gr.Textbox(
                    label="AIS Immune Response Log",
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