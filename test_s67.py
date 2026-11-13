import json
from pathlib import Path
from data.clause_graph_builder import build_graph

raw_path = Path(r"D:\program\artificial intelligence\Fujitsu-Assessing-Compliance-in-Enterprise-Dataset-main\train.json")
with open(raw_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

s67 = data.get("scenarios", [])[67]
g = build_graph(s67, include_same_section=False)
print(f"Sample 67 Edges: {g.edges(data=True)}")
print(f"Clauses: {list(s67['clauses'].keys())}")
for k, v in s67['clauses'].items():
    print(f"{k}: {v[:100]}...")
