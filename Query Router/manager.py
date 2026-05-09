import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from state import AgentState
from models.llm_loader import get_llm
from supervisor import robust_json_extract  # Or duplicate the helper here

_DIR = os.path.dirname(os.path.abspath(__file__))

class QueryDecomposition(BaseModel):
    is_complex: bool = Field(description="True if the question asks for multiple distinct things. False if it is a single question.")
    sub_queries: list[str] = Field(description="A list of strings containing the sub-questions.")

def manager_node(state: AgentState):
    query = state["question"]
    
    with open(os.path.join(_DIR, "prompts", "manager_prompt.txt"), "r") as f:
        manager_prompt = f.read()

    prompt = ChatPromptTemplate.from_template(manager_prompt)
    chain = prompt | get_llm() | StrOutputParser()
    raw = chain.invoke({"question": query})

    result = robust_json_extract(raw, {
        "is_complex": False,
        "sub_queries": [query]
    })

    sub_queries = result.get("sub_queries", [query])
    if not isinstance(sub_queries, list):
        sub_queries = [query]

    return {"sub_queries": sub_queries}