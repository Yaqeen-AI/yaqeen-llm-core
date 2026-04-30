# Query Router Router

This is a LangGraph-based multi-agent routing pipeline. It uses a supervisor agent to route queries to specialized domains (Quran, Hadith, Fiqh), retrieves data, grades the context using a cross-encoder, compresses the context to save VRAM, and synthesizes a final answer.

## Hardware Requirements
- **GPU:** NVIDIA GPU with at least 8GB+ VRAM (required for loading Qwen 2.5 3B).
- **RAM:** 16GB+ System RAM.
- **Disk Space:** ~15GB free for model weights.

## Setup Instructions

**1. Clone or Unzip the repository**
Navigate to the project folder in your terminal.

**2. Create a Virtual Environment (Recommended)**

python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate


# **Query Router simple diagram**


                    User Query
                            ↓
                    Semantic Cache
                      ├─ hit → return saved answer
                      └─ miss → Receptionist / Supervisor
                                           ↓
                            Simple or Complex?
                            ├─ simple → direct route
                            └─ complex → Manager
                                                     ↓
                                          Task Decomposition
                                                      ↓
                            ┌─────────────────────────┼──────────────────────────┐
                            ↓                         ↓                          ↓
                         Worker A                  Worker B                   Worker C
                       (Quran RAG)               (Hadith RAG)               (Fiqh RAG)
                            └─────────────────────────┼──────────────────────────┘
                                                      ↓
                                            Inspector / Reranker
                                          ├─ reject → re-search
                                          └─ approve → Context Compression
                                                      ↓
                                               Writer / Synthesis
                                                       ↓
                                                  Final Answer


# **How our Query Router works?**

**Step 1 — Semantic Caching (The "FAQ" Board):**

    Before processing begins, the system checks whether the exact same question has already been answered recently.

**How it works:**

    If a match exists, the saved response is fetched instantly from system memory.

**Why it matters:**

    It skips the entire pipeline. No model loading, no searching, no writing. It takes milliseconds and uses almost zero GPU memory.

**Step 2 — The Receptionist (Supervisor Agent):**

    If the question is new, it enters a lightweight routing model that analyzes the request type.

**How it works:**

    This agent decides which specific processing path, tools, or expert domain models are required to find the answer.

**Step 3 — KV Cache Offloading (The "Waiting Room"):**

**How it works:**

    While the query is being analyzed, the AI's temporary attention memory (KV Cache) is parked in the 32GB of system RAM instead of being deleted when the Receptionist unloads.

**Why it matters:**

    It drastically speeds up the handover between different agents because they can share their "train of thought" and don't have to re-read the original question from scratch.

**Step 4 — The Manager (Task Decomposition):**

**How it works:**

    If the user asks a complicated, multi-part question, this manager agent slices the prompt into smaller, independent sub-tasks. Single-purpose queries bypass this step entirely.

**Step 5 — The Workers (Specialized Agents):**

**How it works:**

    Each sub-task is routed to the appropriate expert agent (e.g., a SQL Agent or a Document Agent). These workers go out and collect the raw, relevant information from their specific databases and indexes.

**Step 6 — The Inspector (Reranker & Quality Validation):**

**How it works:**

    Before any final text is generated, a quality control model intercepts the fetched data and grades it against the original question.

**Why it matters:**

    If the data is irrelevant, it rejects it and forces the Workers to search again. If it is highly relevant, it approves it. This prevents the system from hallucinating.

**Step 7 — Context Compression (The "Executive Summary"):**

**How it works:**

    Even approved information can be too lengthy for the final model to process. A fast algorithm condenses the text, removing filler words and redundant content while preserving the hardcore facts.

**Why it matters:**

    It keeps the final prompt small, which is critical for preventing out-of-memory crashes.

**Step 8 — The Writer (Synthesis Agent):**

**How it works:**

    The final generation model is loaded onto the GPU. It combines all the validated, compressed information from the Workers and synthesizes it into a single, coherent, and highly accurate response.
