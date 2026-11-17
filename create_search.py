import json, os, random
os.chdir('d:\\program\\artificial intelligence')

CONDITIONS = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']
LABELS = ['Compliant', 'Non-Compliant', 'Not-Applicable']
roles = ['drafter', 'reviewer', 'approver', 'signatory']

sid = "tr" + str(random.randint(1000, 9999)).zfill(4)
condition = random.choice(CONDITIONS)
answer = random.choice(LABELS)
n_clauses = random.randint(10, 30)

# Simple tree
tree = [{"tid": i, "parent": None if i == 0 else i-1, "children": [], "depth": i, "node_id": "n" + str(i), "clause": "clause_" + str(random.randint(1, n_clauses)).zfill(3), "role": random.choice(roles), "action": {"src": "c" + str(random.randint(1, n_clauses)), "edge_type": "refers_to", "dst": "c" + str(random.randint(1, n_clauses))}, "g": round(random.uniform(0, 5), 2), "h": round(random.uniform(0, 5), 2), "f": round(random.uniform(0, 10), 2), "visits": random.randint(1, 100), "value": round(random.uniform(0, 1), 3), "conf": round(random.uniform(0.1, 1.0), 3), "label": random.choice(LABELS), "status": random.choice(["expanded", "frontier", "pruned", "goal", "deadend"]), "flags": [f"flag_{j}" for j in range(random.randint(0, 3))]} for i in range(5)]

# Simple trace
trace = [{"action": {"src": "clause_" + str(random.randint(1, n_clauses)).zfill(3), "edge_type": random.choice(["refers_to", "contradicts", "supports"]), "dst": "clause_" + str(random.randint(1, n_clauses)).zfill(3)}, "depth": random.randint(0, 2), "g": round(random.uniform(0, 5), 2), "h": round(random.uniform(0, 5), 2), "f": round(random.uniform(0, 10), 2), "label": random.choice(LABELS), "conf": round(random.uniform(0.1, 1.0), 3), "clauses": ["clause_" + str(random.randint(1, n_clauses)).zfill(3) for _ in range(random.randint(1, 2))], "clf_score": round(random.uniform(0, 1), 3) if random.choice([True, False]) else None, "llm_reward": round(random.uniform(0, 1), 3) if random.choice([True, False]) else None, "rule_mask": [f"rule_{i}" for i in range(random.randint(1, 2))] if random.choice([True, False]) else None, "adjust": round(random.uniform(-1, 1), 2) if random.choice([True, False]) else None, "expanded": random.choice([True, False])} for _ in range(3)]

# Simple path
path = [{"src": "clause_" + str(random.randint(1, n_clauses)).zfill(3), "edge_type": random.choice(["refers_to", "contradicts", "supports"]), "dst": "clause_" + str(random.randint(1, n_clauses)).zfill(3)} for _ in range(3)]

# Simple meta
meta = {"sid": "tr" + str(random.randint(1000, 9999)).zfill(4), "condition": condition, "gold": answer, "pred": answer, "steps": random.randint(3, 8), "expanded": random.randint(10, 50), "confidence": round(random.uniform(0.5, 0.99), 3), "llm_calls": random.randint(1, 10), "n_clauses": n_clauses, "scenario": "sim_1", "path": [{"src": "c" + str(random.randint(1, n_clauses)), "edge_type": "refers_to", "dst": "c" + str(random.randint(1, n_clauses))} for _ in range(2)], "n_relevant": random.randint(2, 5)}

# Simple explanation
explanation = {"text": "基于证据链分析，判定为 " + answer, "steps": [{"i": i, "from": "clause_" + str(random.randint(1, n_clauses)).zfill(3), "to": "clause_" + str(random.randint(1, n_clauses)).zfill(3), "edge": random.choice(["refers_to", "contradicts", "supports"]), "clause": "clause_" + str(random.randint(1, n_clauses)).zfill(3), "role": random.choice(roles), "excerpt": "证据 " + str(i), "g": round(random.uniform(0, 3), 2), "h": round(random.uniform(0, 3), 2), "f": round(random.uniform(0, 6), 2), "conf": round(random.uniform(0.5, 1.0), 3), "llm_reward": round(random.uniform(0, 1), 3), "clf_score": round(random.uniform(0, 1), 3) if random.choice([True, False]) else None, "rule_mask": [f"rule_{j}" for j in range(random.randint(1, 2))] if random.choice([True, False]) else None, "note": "步骤 " + str(i) + " 备注", "llm_note": "LLM " + str(i) if random.choice([True, False]) else None} for i in range(2)], "citations": [{"clause": "clause_" + str(random.randint(1, n_clauses)).zfill(3), "excerpt": "引文 " + str(i)} for i in range(1)], "signals": {"llm_reward": round(random.uniform(0, 1), 3), "clf_score": round(random.uniform(0, 1), 3) if random.choice([True, False]) else None, "risk": random.randint(1, 3), "rule_hits": {f"rule_{i}": random.randint(0, 3) for i in range(1)}, "expanded_nodes": random.randint(3, 10), "pruned_by_rule": random.randint(0, 2), "llm_rejected": random.randint(0, 1), "llm_calls": random.randint(1, 5)}, "sufficiency": {"covered_clauses": ["clause_" + str(i).zfill(3) for i in range(random.randint(3, 6))], "relevant_clauses": ["clause_" + str(i).zfill(3) for i in range(random.randint(2, 4))], "coverage_ok": random.choice([True, False]), "exception_required": random.choice([True, False]), "exception_ok": random.choice([True, False]), "confidence": round(random.uniform(0.5, 1.0), 3), "tau_conf": round(random.uniform(0.5, 1.0), 3), "confidence_ok": random.choice([True, False]), "reached_goal": random.choice([True, False])}, "mode": random.choice(["template", "real", "auto"]), "violation": random.choice([True, False]), "llm_calls": random.randint(1, 10), "raw": None}

search = {
    "sid": sid,
    "condition": condition,
    "answer": answer,
    "confidence": round(random.uniform(0.5, 0.99), 3),
    "steps": random.randint(3, 8),
    "expanded_nodes": random.randint(10, 50),
    "reached_goal": random.choice([True, False]),
    "llm_calls": random.randint(1, 10),
    "pruned_by_rule": random.randint(0, 2),
    "llm_rejected": random.randint(0, 1),
    "llm_mode": random.choice(["real", "mock"]),
    "tree": tree,
    "n_nodes_raw": random.randint(50, 150),
    "trace": trace,
    "path": path,
    "meta": meta,
    "explanation": explanation,
    "elapsed_ms": random.randint(500, 2000)
}

with open("mock_search.json", "w", encoding="utf-8") as f:
    json.dump(search, f, ensure_ascii=False, indent=2)
print("mock_search.json generated, sid=" + search["sid"] + ", condition=" + search["condition"])