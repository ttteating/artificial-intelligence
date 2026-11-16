"""导出 facet 级条款图的三元组，并随机抽 20 条边供人工抽查。

为什么需要
----------
课程要求"自带实体、关系、三元组"的知识图谱；项目文档 §4 的验收项是
"`graph.json` 可加载，人工抽查 20 条边，记录边类型分布和覆盖率"。
本项目的图是**运行时构建**的（`data/facet_graph.py`，GRAPH_VERSION=facet-v1），
此前只有 B 侧 5 条冒烟的 `data/graphs.json`，没有可交付的全量三元组。
本脚本把 dev（或 test）全量图落成 jsonl，并打印 20 条随机边 + 生成它们的依据，
人工抽查直接对着这份输出看。

用法：
    python -m tools.export_triples                    # dev 全量
    python -m tools.export_triples --split test
    python -m tools.export_triples --split dev --limit 100 --sample 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.clause_graph import EDGE_TYPES
from data.corpus import Corpus

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=20, help="抽样边数（人工抽查用）")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    c = Corpus().load()
    sids = c.ids(args.split)
    if args.limit:
        sids = sids[: args.limit]

    out_path = RESULTS / f"graphs_{args.split}.jsonl"
    RESULTS.mkdir(parents=True, exist_ok=True)

    all_triples: list[dict] = []
    type_counts = {t: 0 for t in EDGE_TYPES}
    with out_path.open("w", encoding="utf-8") as f:
        for sid in sids:
            g = c.graph(sid)
            nodes = [{
                "nid": nid,
                "clause": n.clause,
                "role": n.role,
                "text": n.text[:300],
            } for nid, n in g.nodes.items()]
            triples = []
            for src, outs in g.out.items():
                for e in outs:
                    triples.append({"src": e.src, "etype": e.etype, "dst": e.dst,
                                    "w": round(float(e.w), 4)})
                    type_counts[e.etype] = type_counts.get(e.etype, 0) + 1
            all_triples.extend([{"sid": sid, **t} for t in triples])
            f.write(json.dumps({"sid": sid, "version": g.meta.get("version"),
                                "entry": list(g.entry), "nodes": nodes,
                                "triples": triples}, ensure_ascii=False) + "\n")

    print(f"[export] {len(sids)} 场景 -> {out_path}")
    print(f"[export] 三元组 {len(all_triples)} 条, 边类型分布: "
          + ", ".join(f"{k}={type_counts.get(k, 0)}" for k in EDGE_TYPES))

    # ---- 人工抽查：随机 20 条边，附带两端节点的角色/文本片段
    rng = random.Random(args.seed)
    sample = rng.sample(all_triples, min(args.sample, len(all_triples)))
    md = [f"# 边人工抽查（{args.split}，随机 {len(sample)} 条）", "",
          "抽检要点：① 边类型与两端文本的语义是否匹配；② `src/dst` 是否落在同一场景的条款上；"
          "③ `w`（边权 = 目标条款与场景的相似度）是否随相关度变化。", "",
          "| # | sid | src | etype | dst | w | src 角色/文本片段 | dst 角色/文本片段 |",
          "|---:|---|---|---|---|---:|---|---|"]
    for i, t in enumerate(sample, 1):
        g = c.graph(t["sid"])
        src_n = g.nodes.get(t["src"])
        dst_n = g.nodes.get(t["dst"])
        s_txt = f"{src_n.role}: {src_n.text[:60]}" if src_n else "（ROOT）"
        d_txt = f"{dst_n.role}: {dst_n.text[:60]}" if dst_n else "—"
        md.append(f"| {i} | {t['sid']} | {t['src']} | {t['etype']} | {t['dst']} | "
                  f"{t['w']:.3f} | {s_txt} | {d_txt} |")
    md += ["", "判定口径（人工复核时填）：", "",
           "- 语义匹配：边类型与两端文本的语义关系一致（如 EXCEPTIONS 两端确实是\"主句 vs 例外句\"）",
           "- 可加载性：jsonl 每行是一个场景的 `{sid, version, entry, nodes, triples}`，"
           f"共 {len(sids)} 行",]
    (RESULTS / f"edge_sample_{args.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sample] -> {RESULTS / f'edge_sample_{args.split}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
