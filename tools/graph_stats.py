"""条款图统计：节点/边规模、边类型分布、孤立节点占比。

报告里"条款图覆盖率"那一节需要这些数字，而图是运行时按需构建的
（`data/facet_graph.py`，缓存键 = GRAPH_VERSION），所以这里只跑一遍
`Corpus`，统计后直接打印，便于报告引用。

用法：
    python -m tools.graph_stats                  # dev 全量 600 场景
    python -m tools.graph_stats --split test --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from statistics import mean

from data.clause_graph import EDGE_TYPES
from data.corpus import Corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    c = Corpus().load()
    sids = c.ids(args.split)
    if args.limit:
        sids = sids[: args.limit]

    n_nodes, n_edges, n_isolated, n_entries = [], [], [], []
    etype = Counter()
    etype_by_scen = defaultdict(list)
    for sid in sids:
        g = c.graph(sid)
        nodes = len(g.nodes)
        edges = sum(len(v) for v in g.out.values())
        iso = sum(1 for nid in g.nodes if not g.out.get(nid))
        n_nodes.append(nodes)
        n_edges.append(edges)
        n_isolated.append(iso / max(1, nodes))
        n_entries.append(len(g.entry))
        seen = set()
        for outs in g.out.values():
            for e in outs:
                etype[e.etype] += 1
                seen.add(e.etype)
        for t in seen:
            etype_by_scen[t].append(1)

    total_edges = sum(etype.values())
    stats = {
        "split": args.split,
        "n_scenarios": len(sids),
        "avg_nodes": round(mean(n_nodes), 2),
        "avg_edges": round(mean(n_edges), 2),
        "avg_entry_nodes": round(mean(n_entries), 2),
        "isolated_ratio": round(mean(n_isolated), 4),
        "edges_total": total_edges,
        "edge_type_counts": {k: etype[k] for k in EDGE_TYPES},
        "edge_type_share": {k: round(etype[k] / max(1, total_edges), 4) for k in EDGE_TYPES},
        "scenarios_with_type": {k: len(etype_by_scen[k]) for k in EDGE_TYPES},
        "type_coverage": {k: round(len(etype_by_scen[k]) / max(1, len(sids)), 4) for k in EDGE_TYPES},
    }
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    out = f"results/graph_stats_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print("[saved]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
