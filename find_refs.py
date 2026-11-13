import json
from pathlib import Path

raw_path = Path(r"D:\program\artificial intelligence\Fujitsu-Assessing-Compliance-in-Enterprise-Dataset-main\train.json")
with open(raw_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

scenarios = data.get("scenarios", [])
found = 0
for i, s in enumerate(scenarios):
    clauses = s.get("clauses", {})
    keys = list(clauses.keys())
    for k, text in clauses.items():
        for other_k in keys:
            if k == other_k: continue
            # Extract number from other_k
            num = other_k.replace("evidence_", "")
            if f"Section {num}" in text or f"Clause {num}" in text or f"evidence_{num}" in text or f"Article {num}" in text:
                print(f"Sample {i}: {k} refers to {other_k}")
                found += 1
                if found > 10: break
    if found > 10: break
