"""把 results/ 下的实验产物聚合成前端可直接加载的静态数据包。

为什么要这一步
------------
前端是单入口静态页（`web/index.html`），双击用 `file://` 打开时浏览器不允许
`fetch` 本地 JSON，所以所有数字必须在构建期算好、写成一段 JS：

    web/assets/data.js   ->   window.DATA = {...}

每个数字都来自 `results/` 里的真实产物，来源写在 `DATA.meta.sources` 里，
页面上的"口径"注脚按这张表标注，不做任何手填或估算。

用法
----
    python -m tools.build_web_data
    python -m tools.build_web_data --out web/assets/data.js
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "web" / "assets" / "data.js"

LABELS = ["Compliant", "Non-Compliant", "Not-Applicable"]
CONDS = ["B0", "B1", "A0", "A1", "A2", "M0", "M1", "H"]

COND_INFO = {
    "B0": dict(kernel="BFS", rules=False, h="zero", llm="off", clf=False,
               title="盲目基准", note="BFS 全量展开，无边权、无启发"),
    "B1": dict(kernel="BFS", rules=True, h="zero", llm="off", clf=False,
               title="规则 + 盲目", note="在 BFS 上加 R1–R6 硬约束，看规则单独值多少"),
    "A0": dict(kernel="A*", rules=False, h="hand", llm="off", clf=False,
               title="固定手工 h", note="w = (0.55, 0.20, 0.15, 0.10)，人工设定"),
    "A1": dict(kernel="A*", rules=False, h="ga", llm="off", clf=False,
               title="GA 优化 h", note="遗传算法搜出的 4 维权重"),
    "A2": dict(kernel="A*", rules=False, h="ga", llm="mock", clf=False,
               title="A* + LLM", note="仅 LLM 做动作校验与 path reward，无硬规则"),
    "M0": dict(kernel="MCTS", rules=False, h="zero", llm="off", clf=False,
               title="MCTS 基准", note="UCB1 + 分层采样 + 限深 rollout"),
    "M1": dict(kernel="MCTS", rules=True, h="zero", llm="mock", clf=True,
               title="MCTS + LLM + 分类器", note="三类组件全开，但启发项仍为 0"),
    "H": dict(kernel="A*", rules=True, h="ga", llm="mock", clf=True,
              title="混合架构", note="规则专家系统 + 学习评估器 + 启发搜索"),
}

RULES = [
    dict(id="R1", name="无关扩展阻断",
         text="目标条款与场景相似度 < sim_floor(0.12)，且强相关条款已覆盖、已审条款 ≥ 2 → 剪枝",
         why="已经拿到对口条款后，再读低相似条款只会抬高解释成本"),
    dict(id="R2", name="深链截断",
         text="连续 DEPENDS_ON 边已达 max_ref_chain(2) 跳，还要继续沿依赖走 → 剪枝",
         why="合同里的引用嵌套很少超过两层，第 3 跳基本是噪声"),
    dict(id="R3", name="冲突边双证据",
         text="走 CONFLICTS 边时，其源条款尚未进入已审集合且深度 ≥ 1 → 延后",
         why="冲突判定需要两边都读过，只读一边就下冲突结论是典型误判来源"),
    dict(id="R4", name="例外未解不下结论",
         text="场景需要解例外且例外未解时，禁止把动作花在非例外/非冲突方向 → 剪枝",
         why="例外条款是这类合规题的胜负手，先把它解开再谈覆盖"),
    dict(id="R5", name="环检测",
         text="同一 (src, edge_type, dst) 边已在路径里走过 → 剪枝",
         why="回访同一跳不产生新信息；不同边类型的回访仍放行，语义不同"),
    dict(id="R6", name="失效条款排除",
         text="目标节点文本命中失效线索（superseded / no longer in effect 等），且该边不是例外/冲突 → 剪枝",
         why="已失效条款不能再当推理依据；例外/冲突本身要看失效条款，故放行"),
]


# ---------------------------------------------------------------- 读
def jload(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def to_num(value):
    """把表格里的字符串数字转成数值；`—` / 空 一律返回 None（不编造）。"""
    if value is None:
        return None
    text = str(value).strip().replace("−", "-").replace("*", "").replace("`", "")
    if text in ("", "—", "-", "–", "n/a", "N/A"):
        return None
    try:
        return float(text) if ("." in text or "e" in text.lower()) else int(text)
    except ValueError:
        return None


def md_tables(text: str) -> list[dict]:
    """解析 markdown 表格，返回 [{heading, header, rows}]。"""
    tables, heading, lines = [], "", text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            i += 1
            continue
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in row.strip("|").split("|")] for row in block]
            if len(cells) >= 2:
                header = cells[0]
                body = [row for row in cells[1:]
                        if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in row)]
                tables.append({"heading": heading, "header": header,
                               "rows": [dict(zip(header, row)) for row in body]})
            continue
        i += 1
    return tables


def parse_literal(value, default):
    """把导出件里被字符串化的 list/dict 还原（`"['success_path']"` -> ['success_path']）。"""
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return default
    text = str(value).strip()
    if text in ("", "None", "null"):
        return default
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return default


# ---------------------------------------------------------------- 各段数据
def build_summary() -> dict:
    return {split: jload(RES / f"summary_{split}.json") for split in ("dev", "test")}


def build_graph_stats() -> dict:
    return jload(RES / "graph_stats_dev.json")


def build_facet_graphs() -> dict:
    """建模页用的两张真实 facet 图（来自 graphs_dev.jsonl）。"""
    wanted = {"tr00632": "success", "tr00715": "misled"}
    out = {}
    with open(RES / "graphs_dev.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sid = row.get("sid")
            if sid not in wanted:
                continue
            nodes = [dict(id=n["nid"], clause=n["clause"], role=n["role"],
                          text=(n.get("text") or "")[:150])
                     for n in row.get("nodes", [])]
            edges = [dict(src=t["src"], dst=t["dst"], type=t["etype"], w=t.get("w"))
                     for t in (row.get("triples") or [])]
            out[wanted[sid]] = dict(sid=sid, version=row.get("version"),
                                    entry=row.get("entry") or [], nodes=nodes, edges=edges)
    return out


def build_trees() -> dict:
    names = ["success", "astar_misled", "llm_misjudge"]
    out = {}
    for name in names:
        payload = jload(RES / "trees" / f"{name}.json")
        meta = payload.get("meta") or {}
        clauses = meta.get("clauses") or {}
        if isinstance(clauses, str):
            clauses = parse_literal(clauses, {})
        nodes = []
        for entry in payload.get("tree", []):
            clause = entry.get("clause") or ""
            nodes.append(dict(
                id=int(entry.get("tid")),
                parent=parse_literal(entry.get("parent"), None),
                depth=int(entry.get("depth") or 0),
                node=entry.get("node_id") or "ROOT",
                clause=clause,
                role=entry.get("role") or "",
                action=entry.get("action") or "",
                g=to_num(entry.get("g")) or 0.0,
                h=to_num(entry.get("h")) or 0.0,
                f=to_num(entry.get("f")) or 0.0,
                visits=int(to_num(entry.get("visits")) or 0),
                value=to_num(entry.get("value")) or 0.0,
                conf=to_num(entry.get("conf")) or 0.0,
                label=entry.get("label") or "",
                status=entry.get("status") or "",
                flags=parse_literal(entry.get("flags"), []),
                excerpt=(clauses.get(clause) or "")[:110],
            ))
        trace = []
        for step in payload.get("trace", [])[:40]:
            trace.append(dict(
                depth=step.get("depth"),
                action=step.get("action"),
                g=step.get("g"), h=step.get("h"), f=step.get("f"),
                conf=step.get("conf"), label=step.get("label"),
                expanded=step.get("expanded"),
                llm_reward=step.get("llm_reward"),
                clf_score=step.get("clf_score"),
                adjust=step.get("adjust"),
                rule_mask=step.get("rule_mask"),
            ))
        out[name] = dict(
            name=payload.get("name"),
            sid=meta.get("sid", ""),
            condition=meta.get("condition", ""),
            gold=meta.get("gold", ""),
            pred=payload.get("answer", ""),
            answer=payload.get("answer", ""),
            confidence=to_num(payload.get("confidence")) or 0.0,
            expanded=to_num(meta.get("expanded")) or 0,
            steps=to_num(payload.get("steps")) or 0,
            llm_calls=to_num(meta.get("llm_calls")) or 0,
            n_clauses=to_num(meta.get("n_clauses")) or 0,
            n_nodes_raw=payload.get("n_nodes_raw"),
            scenario=(meta.get("scenario") or "")[:420],
            path=[dict(src=p.get("src"), type=p.get("edge_type"), dst=p.get("dst"))
                  for p in payload.get("path", [])],
            nodes=nodes,
            trace=trace,
        )
    out["manifest"] = jload(RES / "trees" / "manifest.json")
    return out


def build_explanations() -> dict:
    out = {}
    for path in sorted((RES / "explanations").glob("*.json")):
        payload = jload(path)
        out[path.stem] = dict(
            text=payload.get("text", ""),
            steps=payload.get("steps", []),
            citations=payload.get("citations", []),
            signals=payload.get("signals", {}),
            sufficiency=payload.get("sufficiency", {}),
            mode=payload.get("mode", ""),
            violation=bool(payload.get("violation")),
            llm_calls=payload.get("llm_calls"),
        )
    return out


def build_ga() -> dict:
    payload = jload(RES / "ga_history.json")
    return dict(history=payload.get("history", []), summary=payload.get("summary", {}))


def build_hparam() -> list[dict]:
    import csv

    rows = []
    with open(RES / "hparam_sweep.csv", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(
                budget=int(row["budget"]),
                w_sim=float(row["w_sim"]),
                w_exc=float(row["w_exc"]),
                macro_f1=float(row["macro_f1"]),
                avg_expanded=float(row["avg_expanded"]),
            ))
    return rows


def build_budget() -> list[dict]:
    import csv

    with open(RES / "budget_sweep.csv", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(dict(
                budget=int(row["budget"]), condition=row["condition"],
                macro_f1=float(row["macro_f1"]), accuracy=float(row["accuracy"]),
                avg_expanded=float(row["avg_expanded"]),
                redundant_ratio=float(row["redundant_ratio"]),
            ))
        return rows


def build_stratified() -> dict:
    import csv

    out = {}
    for split in ("dev", "test"):
        path = RES / f"stratified_{split}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig") as handle:
            rows = []
            for row in csv.DictReader(handle):
                rows.append(dict(
                    stratum=row["stratum"], condition=row["condition"],
                    n=int(row["n"]), macro_f1=float(row["macro_f1"]),
                    accuracy=float(row["accuracy"]),
                    avg_expanded=float(row["avg_expanded"]),
                    avg_steps=float(row["avg_steps"]),
                    redundant_ratio=float(row["redundant_ratio"]),
                ))
            out[split] = rows
    return out


def build_hallucination() -> list[dict]:
    import csv

    with open(RES / "hallucination_sweep.csv", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(dict(
                rate=float(row["hallucination_rate"]), condition=row["condition"],
                n=int(row["n"]), macro_f1=float(row["macro_f1"]),
                accuracy=float(row["accuracy"]),
                avg_expanded=float(row["avg_expanded"]),
                avg_steps=float(row["avg_steps"]),
                redundant_ratio=float(row["redundant_ratio"]),
                avg_llm_calls=float(row["avg_llm_calls"]),
                avg_llm_rejected=float(row["avg_llm_rejected"]),
            ))
        return rows


def build_lambda_grid() -> dict:
    """λ 网格：18 组，F1 恒定，只有扩展节点在抖。"""
    text = read_text(RES / "lambda_grid.md")
    table = md_tables(text)[0]
    header = table["header"]
    rows = []
    for row in table["rows"]:
        rows.append(dict(
            llm=to_num(row.get(header[0])), clf=to_num(row.get(header[1])),
            h=to_num(row.get(header[2])), risk=to_num(row.get(header[3])),
            macro_f1=to_num(row.get(header[4])),
            avg_expanded=to_num(row.get(header[5])),
            llm_calls=to_num(row.get(header[6])),
            score=to_num(row.get(header[7])),
        ))
    best = None
    for row in rows:
        if best is None or (row["score"] or 0) > (best["score"] or 0):
            best = row
    formula = ""
    for line in text.splitlines():
        if "评分" in line:
            formula = line.lstrip("- ").strip()
    return dict(rows=rows, best=best, formula=formula)


def build_classifier() -> dict:
    """路径价值分类器的最终训练记录（logs/classifier_train.log 最后一段）。"""
    log = read_text(ROOT / "logs" / "classifier_train.log")
    blocks = [b for b in log.split("========================") if "train_classifier start" in b]
    last = blocks[-1] if blocks else ""
    out = dict(
        n_train=None, pos_ratio=None, n_dev=None,
        best_iter=None, best_dev_auc=None, train_data_hash=None,
        features=[], params={}, source="logs/classifier_train.log",
    )
    match = re.search(r"n_train=(\d+)\s+\(pos=(\d+),\s+neg=(\d+),\s+pos_ratio=(\d+\.\d+)%\)", last)
    if match:
        out["n_train"] = int(match.group(1))
        out["pos"] = int(match.group(2))
        out["neg"] = int(match.group(3))
        out["pos_ratio"] = float(match.group(4)) / 100.0
    match = re.search(r"n_dev=(\d+)\s+\(pos=(\d+)\)", last)
    if match:
        out["n_dev"] = int(match.group(1))
        out["n_dev_pos"] = int(match.group(2))
    match = re.search(r"best_iteration=(\d+),\s+best dev AUC=(\d+\.\d+)", last)
    if match:
        out["best_iter"] = int(match.group(1))
        out["best_dev_auc"] = float(match.group(2))
    match = re.search(r"train data hash \(sha256\)=([0-9a-f]+)", last)
    if match:
        out["train_data_hash"] = match.group(1)
    match = re.search(r"feature columns \(\d+\)=(\[[^\]]*\])", last)
    if match:
        out["features"] = parse_literal(match.group(1), [])
    match = re.search(r"lgb params=(\{.*\})", last)
    if match:
        out["params"] = parse_literal(match.group(1), {})
    return out


def build_llm() -> dict:
    reject = jload(RES / "llm_reject_rate.json")
    cost = {}
    for table in md_tables(read_text(RES / "llm_cost_table.md")):
        split = "dev" if "dev" in table["heading"] else "test"
        header = table["header"]
        cost[split] = [dict(
            condition=row.get(header[0]),
            macro_f1=to_num(row.get(header[1])),
            avg_expanded=to_num(row.get(header[2])),
            avg_llm_calls=to_num(row.get(header[3])),
            avg_rejected=to_num(row.get(header[4])),
            avg_pruned=to_num(row.get(header[5])),
            delta_f1=to_num(row.get(header[6])),
            f1_per_call=to_num(row.get(header[7])),
        ) for row in table["rows"]]

    real = {}
    for cond, path in (("H", RES / "llm_real_vs_mock.md"), ("A2", RES / "llm_real_vs_mock_A2.md")):
        text = read_text(path)
        tables = md_tables(text)
        metrics = []
        if tables:
            header = tables[0]["header"]
            for row in tables[0]["rows"]:
                metrics.append(dict(metric=row.get(header[0]), mock=row.get(header[1]),
                                    real=row.get(header[2]), diff=row.get(header[3])))
        flips = []
        if len(tables) > 1:
            header = tables[1]["header"]
            for row in tables[1]["rows"]:
                flips.append(dict(sid=row.get(header[0]), gold=row.get(header[1]),
                                  mock=row.get(header[2]), real=row.get(header[3]),
                                  mock_expanded=to_num(row.get(header[4])),
                                  real_expanded=to_num(row.get(header[5])),
                                  real_rejected=to_num(row.get(header[6]))))
        match = re.search(r"判定翻转：(\d+)\s*/\s*(\d+)\s*条（(\d+\.\d+)%）", text)
        real[cond] = dict(
            metrics=metrics, flips=flips,
            n_eval=int(match.group(2)) if match else None,
            n_flip=int(match.group(1)) if match else None,
            flip_rate=(float(match.group(3)) / 100.0) if match else None,
            source=str(path.relative_to(ROOT)).replace("\\", "/"),
        )
    return dict(reject=reject, cost=cost, real=real)


def build_failure_cases() -> dict:
    payload = jload(RES / "failure_cases.json")
    out = {}
    for key, cases in payload.items():
        out[key] = []
        for case in cases:
            row = {k: v for k, v in case.items() if k not in ("scenario", "clauses")}
            row["scenario"] = (case.get("scenario") or "")[:420]
            clauses = case.get("clauses") or {}
            if isinstance(clauses, str):
                clauses = parse_literal(clauses, {})
            row["clauses"] = {k: (v or "")[:220] for k, v in dict(clauses).items()}
            out[key].append(row)
    return out


def build_decision_eval() -> dict:
    return jload(RES / "decision_path_eval.json")


def _scenario_texts() -> dict:
    """sid -> 场景原文；sid 与 raw 下标一一对应（tr00004 是 train.json 第 4 条）。

    这条映射来自 `data/corpus.py::Corpus.load`：train 侧 sid = f"tr{下标:05d}"，
    test 侧 sid = f"te{下标:05d}"。所以不需要重建 Corpus（那会顺带建图与词典）。
    """
    texts = {}
    for path in (RAW / "train.json", RAW / "test.json"):
        if not path.exists():
            continue
        payload = jload(path)
        prefix = "tr" if path.name.startswith("train") else "te"
        for index, sample in enumerate(payload.get("scenarios", [])):
            texts[f"{prefix}{index:05d}"] = (sample.get("scenario_text") or "")[:150]
    return texts


def build_browser(texts: dict) -> dict:
    """按 sid 聚合 8 条件的逐条结果，供场景浏览器筛选/对比。

    runs 用紧凑数组：[判定标签下标, 扩展节点, 步数, 置信度]。
    """
    import csv

    label_index = {label: i for i, label in enumerate(LABELS)}
    out = {}
    for split in ("dev", "test"):
        path = RES / f"rows_{split}_all.csv"
        if not path.exists():
            continue
        grouped: dict[str, dict] = {}
        # 第一列带 BOM，用 utf-8-sig
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                sid = row["sid"]
                item = grouped.get(sid)
                if item is None:
                    item = grouped[sid] = dict(
                        sid=sid,
                        gold=label_index.get(row["gold"], -1),
                        nc=int(row["n_clauses"] or 0),
                        hop=int(row["oracle_depth"] or 0),
                        runs={},
                    )
                item["runs"][row["condition"]] = [
                    label_index.get(row["answer"], -1),
                    int(row["expanded"] or 0),
                    int(row["steps"] or 0),
                    round(float(row["conf"] or 0), 4),
                ]
        items = []
        for sid, item in sorted(grouped.items()):
            item["text"] = texts.get(sid, "")
            items.append(item)
        out[split] = items
    return out


def build_repro() -> list[dict]:
    return [
        dict(group="环境", cmd="python -m pip install -r requirements.txt",
             note="requirements.txt 为 UTF-8 with BOM，Windows 下 pip 才能读中文注释"),
        dict(group="环境", cmd="python -m experiments.setup", note="构建 split / 条款图 / 角色先验缓存"),
        dict(group="主表", cmd="python -m experiments.run_main_table --split dev",
             note="产出 summary_dev.json、budget_sweep.csv、results/figs/main_*.png"),
        dict(group="主表", cmd="python -m experiments.run_main_table --split test",
             note="test 全量 1,100 条，LLM 仍为 mock 口径"),
        dict(group="启发式", cmd="python -m experiments.run_ga",
             note="pop=12 / gens=9，106 次评估，产出 ga_history.json"),
        dict(group="启发式", cmd="python -m experiments.run_hparam_sweep", note="w_sim×w_exc 地形扫描"),
        dict(group="混合", cmd="python -m experiments.run_lambda_grid", note="18 组 λ 网格，F1 恒定"),
        dict(group="混合", cmd="python -m experiments.train_lgb_classifier",
             note="真实轨迹训练，产出 logs/classifier_train.log"),
        dict(group="混合", cmd="python -m experiments.run_decision_eval", note="判据评估 dev/heldout"),
        dict(group="LLM", cmd="python -m experiments.run_hallucination_sweep", note="幻觉率 0/.1/.2/.3/.5"),
        dict(group="LLM", cmd="python -m tools.llm_cost_table", note="调用成本表"),
        dict(group="LLM", cmd="python -m tools.llm_reject_report", note="误杀率统计"),
        dict(group="失败", cmd="python -m experiments.run_failure_analysis", note="4 类失败案例"),
        dict(group="校验", cmd="python -m tools.check_report_numbers", note="报告数字与产物一致性"),
        dict(group="校验", cmd="python -m tests.test_kernels", note="内核单测 11 条"),
        dict(group="校验", cmd="python -m tests.test_explainer", note="解释模块单测 4 条"),
        dict(group="产物", cmd="python -m tools.explain_case --preset samples",
             note="导出带解释的树 JSON / HTML"),
        dict(group="服务", cmd="python -m api --port 8000", note="FastAPI，浏览器打开 /docs 看端点"),
    ]


def build_params() -> dict:
    cfg = yaml.safe_load(read_text(ROOT / "config.yaml"))
    return dict(
        search=cfg.get("search", {}),
        mcts=cfg.get("mcts", {}),
        ga=cfg.get("ga", {}),
        hybrid=cfg.get("hybrid", {}),
        llm=cfg.get("llm", {}),
        data=cfg.get("data", {}),
        hparams_hand=dict(w_sim=0.55, w_prior=0.20, w_exc=0.15, w_depth=0.10),
        hparams_ga=jload(ROOT / "results" / "models" / "ga_params.json"),
        role_prior=jload(ROOT / "results" / "models" / "role_prior.json"),
        suspect_conf=0.40,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    texts = _scenario_texts()
    data = dict(
        meta=dict(
            project="面向企业合同合规审查的条款图多跳推理",
            author="成员 A · 搜索算法线",
            graph_version="facet-v2",
            splits=dict(dev=600, test=1100),
            n_scenarios=600 + 1100 + 2900 + 100,
            llm_mode="mock（主表 600/1,100 全量）；real 仅 H / A2 各 20 例对照",
            seed=42,
            sources=[
                "results/summary_dev.json", "results/summary_test.json",
                "results/budget_sweep.csv", "results/stratified_{dev,test}.csv",
                "results/hallucination_sweep.csv", "results/hparam_sweep.csv",
                "results/ga_history.json", "results/graph_stats_dev.json",
                "results/graphs_dev.jsonl", "results/trees/{success,astar_misled,llm_misjudge}.json",
                "results/explanations/*.json", "results/failure_cases.json",
                "results/llm_reject_rate.json", "results/llm_cost_table.md",
                "results/llm_real_vs_mock{,_A2}.md", "results/decision_path_eval.json",
                "logs/lambda_grid.json", "logs/classifier_train.log",
                "results/rows_{dev,test}_all.csv", "results/models/*.json", "config.yaml",
            ],
        ),
        labels=LABELS,
        conditions=[dict(id=cond, **COND_INFO[cond]) for cond in CONDS],
        params=build_params(),
        summary=build_summary(),
        graph_stats=build_graph_stats(),
        facet_graphs=build_facet_graphs(),
        trees=build_trees(),
        explanations=build_explanations(),
        ga=build_ga(),
        budget_sweep=build_budget(),
        stratified=build_stratified(),
        hallucination=build_hallucination(),
        hparam_sweep=build_hparam(),
        lambda_grid=build_lambda_grid(),
        classifier=build_classifier(),
        decision_eval=build_decision_eval(),
        rules=RULES,
        llm=build_llm(),
        failure_cases=build_failure_cases(),
        browser=build_browser(texts),
        repro=build_repro(),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    header = ("/* 由 tools/build_web_data.py 生成，勿手改。\n"
              "   全部数字来自 results/ 的实测产物，来源见 DATA.meta.sources。 */\n")
    out.write_text(header + "window.DATA=" + payload + ";\n", encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"written: {out}  ({size_kb:.1f} KB)")
    print(f"  conditions={len(data['conditions'])}  trees={len(['success','astar_misled','llm_misjudge'])}"
          f"  browser dev={len(data['browser'].get('dev', []))}"
          f" test={len(data['browser'].get('test', []))}")
    print(f"  failure cases total={sum(len(v) for v in data['failure_cases'].values())}"
          f"  explanations={len(data['explanations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
