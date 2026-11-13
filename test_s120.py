import json
import re
from pathlib import Path
from data.clause_graph_builder import build_graph, extract_target_id

raw_path = Path(r"D:\program\artificial intelligence\Fujitsu-Assessing-Compliance-in-Enterprise-Dataset-main\train.json")
with open(raw_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

sample_120 = data.get("scenarios", [])[120]
g = build_graph(sample_120, include_same_section=False)
print(f"Sample 120 Edges: {g.edges(data=True)}")
print(f"Unresolved: {g.graph.get('unresolved_count')}")

for k, v in sample_120['clauses'].items():
    print(f"{k}: {v[:100]}...")
