import torch
import gc
from state import AgentState

def memory_management_node(state: AgentState):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return state