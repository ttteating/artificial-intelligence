import json
import networkx as nx
import json
import re
from pathlib import Path
from data.clause_graph_builder import build_graph, extract_target_id, ANCHORS

raw_path = Path(r"D:\program\artificial intelligence\Fujitsu-Assessing-Compliance-in-Enterprise-Dataset-main\train.json")
with open(raw_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

sample_28 = data.get("scenarios", [])[28]
clauses = sample_28['clauses']

for src_id, src_text in clauses.items():
    for edge_type, anchor_list in ANCHORS.items():
        for anchor_pattern, confidence in anchor_list:
            pattern = f"\\b({anchor_pattern})\\b\\s+([^;,]{{1,60}})"
            matches = re.finditer(pattern, src_text, re.IGNORECASE)
            for match in matches:
                anchor_text = match.group(1)
                context = match.group(2)
                targets = extract_target_id(context)
                print(f"Match in {src_id}: '{anchor_text}' context: '{context}' targets: {targets}")
