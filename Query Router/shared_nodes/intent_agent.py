import os
import sys
import json
import logging

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DIR)



logger = logging.getLogger("shared.intent_agent")

def intent_agent_node(state: dict) -> dict:
    """
    Dynamic K Selection Intent Agent.
    Analyzes each sub-query independently and assigns a K value per worker based on user intent.
    Placed after Manager (Task Decomposition) and before Dispatcher/Workers.
    """
    print("   [Intent Agent] -> Analyzing sub-queries for dynamic K selection...")
    sub_queries = state.get("sub_queries", [])
    
    sub_query_k_map = {}
    
    if not sub_queries:
        return {"sub_query_k_map": {}}

    prompt = f"""
You are the Intent Analysis Agent for Yaqeen.
Analyze the following sub-queries and assign dynamic K values for the retrieval workers (quran, hadith, fiqh).

K Selection Rules:
- Very specific query (single ruling, single verse, single hadith) → K = 5
- Moderate query (topic-based, comparative) → K = 10
- Broad/narrative query (full stories, comprehensive topics) → K = 20

You must return a raw JSON object (without markdown blocks) mapping each sub-query text exactly to its intent and K values.
Format:
{{
  "sub_query_text_here": {{
    "intent": "specific" | "moderate" | "broad",
    "k_quran": 5,
    "k_hadith": 5,
    "k_fiqh": 5
  }}
}}

Sub-queries to analyze:
{json.dumps(sub_queries, ensure_ascii=False)}

Output ONLY valid JSON:
"""
    try:
        from shared_nodes.models.llm_loader import get_llm
        # Fast inference for intent
        llm = get_llm(reasoning=False)
        response = llm.invoke(prompt)
        
        if hasattr(response, "content"):
            response = response.content
            
        from nsa import strip_think_tags
        response = strip_think_tags(response)
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        parsed_map = json.loads(response.strip())
        
        # Validation and mapping
        for sq in sub_queries:
            if sq in parsed_map and isinstance(parsed_map[sq], dict):
                sq_data = parsed_map[sq]
                intent = str(sq_data.get("intent", "moderate")).lower()
                
                # Assign fallback based on intent if missing or malformed
                default_k = 5 if intent == "specific" else 15 if intent == "broad" else 8
                
                # Ensure all required keys exist and are integers
                try:
                    k_quran = int(sq_data.get("k_quran", default_k))
                    k_hadith = int(sq_data.get("k_hadith", default_k))
                    k_fiqh = int(sq_data.get("k_fiqh", default_k))
                except (ValueError, TypeError):
                    k_quran = k_hadith = k_fiqh = default_k

                sub_query_k_map[sq] = {
                    "intent": intent,
                    "k_quran": k_quran,
                    "k_hadith": k_hadith,
                    "k_fiqh": k_fiqh
                }
            else:
                # If key missing or not a dict, fallback to moderate defaults
                sub_query_k_map[sq] = {"intent": "moderate", "k_quran": 8, "k_hadith": 8, "k_fiqh": 8}
                
        for sq, vals in sub_query_k_map.items():
            intent = vals.get("intent", "moderate").upper()
            kq = vals.get("k_quran", 8)
            kh = vals.get("k_hadith", 8)
            kf = vals.get("k_fiqh", 8)
            print(f"      [{intent}] '{sq[:40]}...' -> Q:{kq} H:{kh} F:{kf}")

    except Exception as e:
        print(f"   [Intent Agent] -> WARNING: Intent parsing failed ({e}). Defaulting to K=8 (moderate).")
        for sq in sub_queries:
            sub_query_k_map[sq] = {"intent": "moderate", "k_quran": 8, "k_hadith": 8, "k_fiqh": 8}

    return {"sub_query_k_map": sub_query_k_map}
