import json
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from state import AgentState
from models.llm_loader import llm

class RouteDecision(BaseModel):
    next_node: str = Field(description="Exact agent name: 'quran_agent', 'hadith_agent', 'fiqh_agent', or 'direct_answer'.")
    reasoning: str = Field(description="Brief explanation.")

def supervisor_node(state: AgentState):
    query = state["question"]
    
    with open("prompts/supervisor_prompt.txt", "r") as f:
        supervisor_prompt = f.read()

    prompt = ChatPromptTemplate.from_template(supervisor_prompt)

    def robust_json_extract(text: str, fallback: dict) -> dict:
        if not text or not isinstance(text, str): return fallback
        if "```json" in text: text = text.split("```json")[-1].split("```")[0]
        elif "```" in text: text = text.split("```")[-1].split("```")[0]
        text = text.strip()
        if not text: return fallback
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1: return fallback
        clean = text[start:end+1]
        clean = re.sub(r",\s*}", "}", clean)
        clean = re.sub(r",\s*]", "]", clean)
        try: return json.loads(clean)
        except: return fallback

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": query})
    result = robust_json_extract(raw, {"next_node": "quran_agent"})
    
    chosen = result.get("next_node", "quran_agent")
    if chosen not in ("quran_agent", "hadith_agent", "fiqh_agent", "direct_answer"):
        chosen = "quran_agent"

    return {
        "current_agent": chosen,
        "loop_step": state.get("loop_step", 0) + 1
    }