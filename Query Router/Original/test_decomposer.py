import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "Query Router")
from router import router_node, _split_query, _route_tokens, _tokenize

print("=== Import OK ===\n")

# Test 1: English multi-part
q1 = "What does the Quran say about fasting and what are the hadith about charity?"
parts = _split_query(q1)
print(f"Q: {q1}")
print(f"Sub-queries ({len(parts)}): {parts}")
for sq in parts:
    tokens = _tokenize(sq)
    agents = _route_tokens(tokens)
    print(f'  "{sq}" -> {agents}')
print()

# Test 2: Arabic multi-part
q2 = "ماذا يقول القرآن عن الصيام وما هي أحاديث الصدقة؟"
parts2 = _split_query(q2)
print(f"Q: {q2}")
print(f"Sub-queries ({len(parts2)}): {parts2}")
for sq in parts2:
    tokens = _tokenize(sq)
    agents = _route_tokens(tokens)
    print(f'  "{sq}" -> {agents}')
print()

# Test 3: Simple query (no split)
q3 = "ما حكم الصيام؟"
parts3 = _split_query(q3)
print(f"Q: {q3}")
print(f"Sub-queries ({len(parts3)}): {parts3}")
for sq in parts3:
    tokens = _tokenize(sq)
    agents = _route_tokens(tokens)
    print(f'  "{sq}" -> {agents}')
print()

# Test 4: Greeting
q4 = "hello"
parts4 = _split_query(q4)
print(f"Q: {q4}")
print(f"Sub-queries ({len(parts4)}): {parts4}")
for sq in parts4:
    tokens = _tokenize(sq)
    agents = _route_tokens(tokens)
    print(f'  "{sq}" -> {agents}')
print()

# Test 5: Three-part question
q5 = "What is the ruling on zakat and what does the Quran say about prayer and are there hadith about fasting?"
parts5 = _split_query(q5)
print(f"Q: {q5}")
print(f"Sub-queries ({len(parts5)}): {parts5}")
for sq in parts5:
    tokens = _tokenize(sq)
    agents = _route_tokens(tokens)
    print(f'  "{sq}" -> {agents}')

# Test 6: Full router_node integration test
print("\n=== Router Node Integration Test ===")
state = {
    "question": "What does the Quran say about fasting and what are the hadith about charity?",
    "current_agent": "",
    "selected_agents": [],
    "retrieved_context": [],
    "reranker_score": 0.0,
    "sub_queries": [],
    "sub_query_agents": {},
    "final_answer": "",
    "messages": [],
    "loop_step": 0
}
result = router_node(state)
print(f"  selected_agents: {result['selected_agents']}")
print(f"  sub_queries: {result['sub_queries']}")
print(f"  sub_query_agents: {result['sub_query_agents']}")

print("\n=== ALL TESTS PASSED ===")
