import json, os, random
os.chdir('d:\\program\\artificial intelligence')

CONDITIONS = ['B0', 'B1', 'A0', 'A1', 'A2', 'M0', 'M1', 'H']
LABELS = ['Compliant', 'Non-Compliant', 'Not-Applicable']

def gen_overview():
    conditions_info = []
    for _ in range(random.randint(3, 8)):
        ci = {
            "id": random.choice(CONDITIONS),
            "kernel": random.choice(["b0_kernel", "ah_kernel"]),
            "rules": random.choice([True, False]),
            "h": f"{random.uniform(0.1, 5.0):.2f}",
            "llm": random.choice(["real", "mock"]),
            "clf": random.choice([True, False]),
            "description": "条件 " + random.choice(CONDITIONS) + " 描述"
        }
        conditions_info.append(ci)
    summary_dev = {}
    for split_key in ['H', 'B0', 'A0', 'M0']:
        summary_dev[split_key] = {
            "n": random.randint(50, 500),
            "macro_f1": round(random.uniform(0.3, 0.95), 3),
            "accuracy": round(random.uniform(0.6, 0.98), 3),
            "avg_expanded": round(random.uniform(5, 50), 2),
            "avg_steps": random.randint(3, 15),
            "redundant_ratio": round(random.uniform(0.1, 0.6), 3),
            "avg_llm_calls": random.randint(1, 20),
            "goal_rate": round(random.uniform(0.4, 0.9), 3),
            "avg_pruned": random.randint(1, 10),
            "avg_llm_rejected": random.randint(0, 5),
            "recall_Compliant": round(random.uniform(0.5, 0.95), 3),
            "recall_Non-Compliant": round(random.uniform(0.4, 0.9), 3),
            "recall_Not-Applicable": round(random.uniform(0.3, 0.8), 3),
            "confusion": {
                "Compliant": {"Compliant": random.randint(30, 80), "Non-Compliant": random.randint(5, 25), "Not-Applicable": random.randint(2, 15)},
                "Non-Compliant": {"Compliant": random.randint(5, 20), "Non-Compliant": random.randint(30, 70), "Not-Applicable": random.randint(3, 15)},
                "Not-Applicable": {"Compliant": random.randint(2, 10), "Non-Compliant": random.randint(3, 12), "Not-Applicable": random.randint(20, 60)}
            }
        }
    trees = {}
    for tree_name in ['success', 'astar_misled', 'llm_misjudge']:
        trees[tree_name] = {
            "sid": "tr" + str(random.randint(1000, 9999)).zfill(4),
            "condition": random.choice(CONDITIONS),
            "json": json.dumps({}, ensure_ascii=False),
            "html": "模拟搜索树",
            "nodes": random.randint(8, 25)
        }
    health = {
        "status": "ok",
        "ready": True,
        "uptime_sec": random.randint(3600, 86400),
        "n_scenarios": random.randint(100, 1000),
        "llm_mode": random.choice(["real", "mock"]),
        "graph_version": "v2.1",
        "error": None
    }
    result = {"conditions": conditions_info, "summary_dev": summary_dev, "graph_stats": {"density": round(random.uniform(0.1, 0.5), 3)}, "trees": trees, "llm_reject": {"reference": random.randint(50, 200), "rows": []}, "ga_summary": {"best_f1": round(random.uniform(0.7, 0.95), 3)}, "lambda_best": {"lambda": round(random.uniform(0.1, 0.9), 2)}, "counts": {"dev": random.randint(100, 500), "test": random.randint(50, 300)}, "health": health}

overview = gen_overview()
with open("mock_overview.json", "w", encoding="utf-8") as f:
    json.dump(overview, f, ensure_ascii=False, indent=2)
print("mock_overview.json generated")