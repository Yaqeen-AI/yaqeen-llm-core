import gc
from state import AgentState

try:
    import torch
except Exception:
    torch = None

def memory_management_node(state: AgentState):
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return state