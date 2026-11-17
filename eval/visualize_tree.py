"""搜索树可视化：导出 JSON + 自包含 HTML（内嵌 SVG，无需 pyvis/graphviz）。

节点配色（对齐"搜索树可视化样例"的要求）：
  绿 = 成功路径 / 目标     灰 = 冗余分支
  紫 = 被规则剪枝          橙 = 被 LLM 判为非法（误杀）
  蓝 = 已扩展              白 = 前沿未展开
边按类型着色：EXCEPTIONS 红、CONFLICTS 橙、DEPENDS_ON 蓝、DEFINES 青、
COMPLEMENTS 灰、SEQUENCE 浅灰、CITES 淡紫。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

EDGE_COLORS = {
    "EXCEPTIONS": "#e03131", "CONFLICTS": "#f08c00", "DEPENDS_ON": "#1c7ed6",
    "DEFINES": "#0ca678", "COMPLEMENTS": "#868e96", "SEQUENCE": "#ced4da",
    "CITES": "#b197fc", "ENTRY": "#adb5bd",
}

ROLE_COLORS = {
    "OBL": "#1c7ed6", "EXC": "#e03131", "COND": "#f08c00",
    "DEF": "#0ca678", "SCOPE": "#7048e8", "REMEDY": "#c2255c",
}


def _node_style(n: dict) -> tuple[str, str]:
    flags = n.get("flags", [])
    if "success_path" in flags or n.get("status") == "goal":
        return "#d3f9d8", "#2f9e44"
    if "rule_pruned" in flags:
        return "#ede7f6", "#7048e8"
    if "llm_rejected" in flags or "llm_misjudge" in flags:
        return "#fff3bf", "#f08c00"
    if "redundant" in flags:
        return "#f1f3f5", "#adb5bd"
    return "#e7f5ff", "#1c7ed6"


def layout(nodes: list[dict], dx: int = 168, dy: int = 92) -> tuple[dict, int, int]:
    by_tid = {n["tid"]: n for n in nodes}
    children = {n["tid"]: list(n.get("children", [])) for n in nodes}
    pos: dict[int, tuple[float, float]] = {}
    order = {"x": 0}

    def walk(tid: int) -> float:
        ch = children.get(tid, [])
        if not ch:
            x = order["x"]
            order["x"] += 1
        else:
            xs = [walk(c) for c in ch]
            x = sum(xs) / len(xs)
        pos[tid] = (x, by_tid[tid]["depth"])
        return x

    roots = [n["tid"] for n in nodes if n.get("parent") is None]
    for r in roots:
        walk(r)
    return pos, max(1, order["x"]), max(1, max(n["depth"] for n in nodes) + 1)


def prune_frontier(nodes: list[dict]) -> list[dict]:
    """去掉"只入队、未展开"的前沿叶子：它们在树里只是噪声，会让图宽到无法阅读。

    保留：被扩展过的节点、目标/成功路径节点、被剪枝/驳回的节点（这些是分析对象）。
    """
    keep = {n["tid"] for n in nodes
            if n.get("status") != "frontier" or "success_path" in n.get("flags", [])}
    idmap, out = {}, []
    for n in nodes:
        if n["tid"] not in keep:
            continue
        m = dict(n)
        m["tid"] = len(out)
        idmap[n["tid"]] = m["tid"]
        out.append(m)
    for m in out:
        m["parent"] = idmap.get(m.get("parent")) if m.get("parent") is not None else None
        m["children"] = [idmap[c] for c in m.get("children", []) if c in idmap]
    # 丢掉父节点被剪掉后的孤儿（理论上不会出现，防御性处理）
    return [m for m in out if m["parent"] is not None or not out or m is out[0]]


def to_svg(nodes: list[dict], title: str = "") -> str:
    pos, n_leaf, n_depth = layout(nodes)
    dx, dy, pad_x, pad_y = 168, 96, 40, 56
    w = int(n_leaf * dx + pad_x * 2)
    h = int(n_depth * dy + pad_y * 2)
    by_tid = {n["tid"]: n for n in nodes}

    def px(tid): return pad_x + pos[tid][0] * dx + dx / 2
    def py(tid): return pad_y + pos[tid][1] * dy

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Inter,Segoe UI,Microsoft YaHei,sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
    ]
    # 边
    for n in nodes:
        for c in n.get("children", []):
            etype = (by_tid[c].get("action") or {}).get("edge_type", "ENTRY")
            color = EDGE_COLORS.get(etype, "#adb5bd")
            x1, y1, x2, y2 = px(n["tid"]), py(n["tid"]) + 18, px(c), py(c) - 20
            my = (y1 + y2) / 2
            parts.append(
                f'<path d="M{x1:.1f},{y1:.1f} C{x1:.1f},{my:.1f} {x2:.1f},{my:.1f} {x2:.1f},{y2:.1f}" '
                f'fill="none" stroke="{color}" stroke-width="1.6" opacity="0.85"/>')
            parts.append(
                f'<text x="{(x1+x2)/2:.1f}" y="{my-2:.1f}" font-size="9" fill="{color}" '
                f'text-anchor="middle">{html.escape(etype)}</text>')
    # 节点
    for n in nodes:
        x, y = px(n["tid"]), py(n["tid"])
        fill, stroke = _node_style(n)
        wbox, hbox = 150, 38
        label = f'{(n.get("role") or "-")} · {n.get("clause") or "ROOT"}'
        sub = f'd{n["depth"]}  g={n["g"]:.0f} h={n["h"]:.2f}'
        if n.get("conf") is not None:
            sub += f'  p={n["conf"]:.2f}'
        tip = (f'state: {n.get("node_id")}  depth={n["depth"]}\n'
               f'g={n["g"]} h={n["h"]} f={n["f"]}\n'
               f'label={n.get("label")} conf={n.get("conf")}\n'
               f'flags={",".join(n.get("flags", [])) or "-"}')
        parts.append(f'<g><title>{html.escape(tip)}</title>')
        parts.append(
            f'<rect x="{x-wbox/2:.1f}" y="{y-hbox/2:.1f}" width="{wbox}" height="{hbox}" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        if n.get("role"):
            rc = ROLE_COLORS.get(n["role"], "#495057")
            parts.append(f'<circle cx="{x-wbox/2+11:.1f}" cy="{y-4:.1f}" r="3.4" fill="{rc}"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y+1:.1f}" font-size="11" fill="#212529" '
            f'text-anchor="middle">{html.escape(label)}</text>')
        parts.append(
            f'<text x="{x:.1f}" y="{y+14:.1f}" font-size="9" fill="#868e96" '
            f'text-anchor="middle">{html.escape(sub)}</text>')
        parts.append('</g>')

    # 图例
    legend = [("成功路径/目标", "#d3f9d8", "#2f9e44"), ("冗余分支", "#f1f3f5", "#adb5bd"),
              ("规则剪枝", "#ede7f6", "#7048e8"), ("LLM 驳回", "#fff3bf", "#f08c00"),
              ("已扩展", "#e7f5ff", "#1c7ed6")]
    lx = pad_x
    for name, f_, s_ in legend:
        parts.append(f'<rect x="{lx}" y="12" width="12" height="12" rx="3" fill="{f_}" stroke="{s_}"/>')
        parts.append(f'<text x="{lx+17}" y="22" font-size="10" fill="#495057">{html.escape(name)}</text>')
        lx += 20 + len(name) * 11

    parts.append('</svg>')
    return "\n".join(parts)


def export(result, out_dir: str | Path, name: str, meta: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = result.tree_root
    nodes = [n.to_dict() for n in rec.nodes] if hasattr(rec, "nodes") else []
    n_raw = len(nodes)
    nodes = prune_frontier(nodes)
    data = {
        "name": name,
        "answer": result.answer,
        "confidence": result.confidence,
        "expanded_nodes": result.expanded_nodes,
        "steps": result.steps,
        "path": [a.to_dict() for a in result.path],
        "trace": result.trace,
        "meta": meta or {},
        "n_nodes_raw": n_raw,
        "tree": nodes,
        # 推理自然语言解释（llm/explainer.py）：默认 None，只有显式生成才有
        "explanation": getattr(result, "explanation", None),
    }
    (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    svg = to_svg(nodes, name)
    trace_rows = "".join(
        f"<tr><td>{i}</td><td>{t.get('clauses')}</td><td>{t.get('label')}</td>"
        f"<td>{t.get('conf')}</td><td>{t.get('g')}</td><td>{t.get('h')}</td>"
        f"<td>{t.get('f')}</td><td>{t.get('llm_reward', '-')}</td>"
        f"<td>{t.get('clf_score', '-')}</td></tr>"
        for i, t in enumerate(result.trace[:60]))

    exp = getattr(result, "explanation", None) or {}
    exp_text = (exp.get("text") or "").strip()
    explain_html = ""
    if exp_text:
        paras = "".join(f"<p>{html.escape(p)}</p>" for p in exp_text.split("\n\n") if p.strip())
        badges = " · ".join(filter(None, [
            f"mode={html.escape(str(exp.get('mode')))}",
            "越权检查：通过" if not exp.get("violation") else "越权检查：<b>命中标签词，已回退模板</b>",
            f"LLM 调用 {exp.get('llm_calls', 0)} 次" if exp.get("llm_calls") else "",
        ]))
        cites = "、".join(c.get("clause", "") for c in (exp.get("citations") or [])[:8])
        explain_html = (
            f'<div class="card"><div style="font-weight:600;margin-bottom:8px">'
            f'推理自然语言解释</div>{paras}'
            f'<div class="sub" style="margin-top:10px">{badges}'
            + (f" · 证据条款：{html.escape(cites)}" if cites else "")
            + "</div></div>"
        )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(name)} · 搜索树</title>
<style>
 body{{margin:0;padding:24px 28px;background:#f8f9fa;color:#212529;
      font:14px/1.6 Inter,"Segoe UI","Microsoft YaHei",sans-serif}}
 h1{{font-size:19px;margin:0 0 4px}} .sub{{color:#868e96;font-size:12px;margin-bottom:16px}}
 .card{{background:#fff;border:1px solid #e9ecef;border-radius:10px;padding:14px;margin-bottom:16px;
       overflow:auto}}
 table{{border-collapse:collapse;font-size:12px;width:100%}}
 th,td{{border-bottom:1px solid #f1f3f5;padding:5px 8px;text-align:left}}
 th{{background:#f8f9fa;font-weight:600}} code{{background:#f1f3f5;padding:1px 5px;border-radius:4px}}
</style></head><body>
<h1>{html.escape(name)}</h1>
<div class="sub">answer=<code>{html.escape(str(result.answer))}</code>
 conf={result.confidence:.3f} · 扩展 {result.expanded_nodes} 节点 · 路径 {result.steps} 步 ·
 {html.escape(json.dumps(meta or {}, ensure_ascii=False))}</div>
<div class="card">{svg}</div>
{explain_html}
<div class="card"><table>
<tr><th>#</th><th>clauses</th><th>label</th><th>conf</th><th>g</th><th>h</th><th>f</th>
<th>llm_reward</th><th>clf_score</th></tr>{trace_rows}</table></div>
</body></html>"""
    (out / f"{name}.html").write_text(page, encoding="utf-8")
    return {"json": str(out / f"{name}.json"), "html": str(out / f"{name}.html"),
            "nodes": len(nodes)}
