import json
import logging
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Regex Anchors
# confidence: 1.0 for very specific legal terms, 0.8 for more general ones
ANCHORS = {
    "cites": [
        (r"as set out in", 1.0),
        (r"pursuant to", 1.0),
        (r"in accordance with", 1.0),
        (r"referred to in", 1.0),
        (r"referenced in", 0.9),
        (r"under", 0.7),
        (r"in", 0.5),
        (r"per", 0.6),
        (r"see", 0.7),
    ],
    "exception_of": [
        (r"subject to", 1.0),
        (r"notwithstanding", 1.0),
        (r"except as", 1.0),
        (r"with the exception of", 1.0),
    ],
    "conditions_on": [
        (r"provided that", 1.0),
        (r"on condition that", 1.0),
        (r"if and only if", 1.0),
        (r"contingent upon", 0.9),
    ],
    "defines": [
        (r"shall mean", 1.0),
        (r"is defined as", 1.0),
        (r"has the meaning", 1.0),
        (r"referred to as", 0.9),
    ],
    "supersedes": [
        (r"supersedes", 1.0),
        (r"replaces", 1.0),
        (r"in lieu of", 1.0),
        (r"prevails over", 0.9),
    ]
}

# Regex to find target references
REF_PATTERNS = [
    r"(?:Section|Clause|evidence_|paragraph|Article)\s+(?:this\s+|the\s+|following\s+)?(\d+(?:\.\d+)*[a-z]?)",
    r"\b(\d+(?:\.\d+)*[a-z]?)\b",
]

def extract_target_id(text: str) -> List[str]:
    """Extracts all potential target IDs from text snippet."""
    results = []
    # Search for specific legal references first
    for pattern in REF_PATTERNS[:-1]:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            results.append(match.group(1))
    
    # If no specific references, try finding any number-like ID
    if not results:
        matches = re.finditer(REF_PATTERNS[-1], text)
        for match in matches:
            results.append(match.group(1))
            
    return list(set(results))

def build_graph(sample: dict, include_same_section: bool = True) -> nx.DiGraph:
    """
    Builds a directed graph of clauses for a single ACE sample.
    """
    clauses = sample.get("clauses", {})
    g = nx.DiGraph()
    
    # Add nodes (keep the clause text on the node so downstream consumers
    # e.g. path_classifier.sem_sim can read it from the graph)
    for evidence_id, evidence_text in clauses.items():
        g.add_node(evidence_id, text=evidence_text)
    
    unresolved_count = 0
    
    # 1. Functional Edges (cites, exception_of, etc.)
    for src_id, src_text in clauses.items():
        for edge_type, anchor_list in ANCHORS.items():
            for anchor_pattern, confidence in anchor_list:
                # Search for anchor + some context
                # Legal text often has "Section 9", "Clause 4", etc.
                pattern = f"\\b({anchor_pattern})\\b\\s+([^;,]{{1,60}})"
                matches = re.finditer(pattern, src_text, re.IGNORECASE)
                
                for match in matches:
                    anchor_text = match.group(1)
                    context = match.group(2)
                    target_vals = extract_target_id(context)
                    
                    if not target_vals:
                        continue

                    for target_val in target_vals:
                        dst_id = None
                        mapping = None

                        # Strategy 1: Direct match "evidence_X" or "X" if evidence_X exists
                        candidates = [
                            f"evidence_{target_val}",
                            f"evidence_{target_val.split('.')[0]}" # Handle 14.3 -> evidence_14
                        ]
                        for cand in candidates:
                            if cand in clauses:
                                dst_id = cand
                                mapping = "id_number"   # 编号直配：启发式，置信按 0.6 折
                                break

                        # Strategy 2: Check if any clause starts with the target section number
                        if not dst_id:
                            # Try to find target_val at the start of other clauses
                            # e.g. target_val is "14.3", check if a clause starts with "14.3" or "Section 14.3"
                            for eid, etext in clauses.items():
                                clean_etext = etext.strip().lower()
                                if clean_etext.startswith(f"section {target_val.lower()}") or \
                                   clean_etext.startswith(f"clause {target_val.lower()}") or \
                                   re.match(rf"^{re.escape(target_val)}\\b", clean_etext):
                                    dst_id = eid
                                    mapping = "heading"
                                    break

                        if dst_id and dst_id != src_id:
                            conf = confidence if mapping == "heading" else confidence * 0.6
                            # Use higher confidence if multiple anchors found
                            if g.has_edge(src_id, dst_id):
                                if g.edges[src_id, dst_id].get('confidence', 0) < conf:
                                    g.add_edge(src_id, dst_id, type=edge_type, anchor_text=anchor_text,
                                               confidence=conf, mapping=mapping)
                            else:
                                g.add_edge(src_id, dst_id, type=edge_type, anchor_text=anchor_text,
                                           confidence=conf, mapping=mapping)
                        else:
                            unresolved_count += 1

    # 2. same_section Edges (Siblings)
    # If two clauses are from the same sample, they are in the same contract.
    # We link them as siblings if they look like they belong to the same section.
    if include_same_section:
        nodes = sorted(list(g.nodes()))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                # Only add same_section if no functional edge exists
                if not g.has_edge(n1, n2):
                    g.add_edge(n1, n2, type="same_section", confidence=0.5,
                               anchor_text="same_contract", mapping="fallback")
                if not g.has_edge(n2, n1):
                    g.add_edge(n2, n1, type="same_section", confidence=0.5,
                               anchor_text="same_contract", mapping="fallback")
    
    g.graph['unresolved_count'] = unresolved_count
    return g

def build_all_graphs(samples: List[dict], output_path: Path) -> None:
    """
    Builds graphs for all samples and saves to graph.json and graph_stats.json.
    """
    all_graphs = {}
    stats = {
        "edge_type_dist": {},
        "mapping_dist": {},
        "total_unresolved": 0,
        "total_edges": 0,
        "functional_edges": 0,
        "total_nodes": 0,
        "sample_count": len(samples),
        "isolated_nodes_ratio": 0.0,
        "clauses_with_functional_edge": 0,
        "edge_coverage": 0.0,
    }

    isolated_nodes_count = 0
    nodes_with_functional = 0

    for i, sample in enumerate(samples):
        # Sample might not have a unique ID, use index or a hash of scenario_text
        sample_id = sample.get("id", str(i))
        g = build_graph(sample)

        # Convert to serializable format
        all_graphs[sample_id] = nx.node_link_data(g)

        # Update stats
        stats["total_unresolved"] += g.graph.get('unresolved_count', 0)
        stats["total_edges"] += g.number_of_edges()
        stats["total_nodes"] += g.number_of_nodes()

        func_nodes = set()
        for u, v, data in g.edges(data=True):
            t = data['type']
            stats["edge_type_dist"][t] = stats["edge_type_dist"].get(t, 0) + 1
            if t != "same_section":
                stats["functional_edges"] += 1
                func_nodes.add(u)
                func_nodes.add(v)
            m = data.get("mapping", "fallback")
            stats["mapping_dist"][m] = stats["mapping_dist"].get(m, 0) + 1

        nodes_with_functional += len(func_nodes)

        for node in g.nodes():
            if g.degree(node) == 0:
                isolated_nodes_count += 1

    if stats["total_nodes"] > 0:
        stats["isolated_nodes_ratio"] = isolated_nodes_count / stats["total_nodes"]
        stats["clauses_with_functional_edge"] = nodes_with_functional
        stats["edge_coverage"] = nodes_with_functional / stats["total_nodes"]

    stats["avg_edges_per_contract"] = stats["total_edges"] / len(samples) if samples else 0

    # Save files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_graphs, f, indent=2)

    stats_path = output_path.parent / "graph_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Built {len(samples)} graphs. Saved to {output_path}")
    logger.info(f"Stats: {stats}")

def audit_graph(graph: nx.DiGraph, sample: dict, n: int = 20) -> None:
    """
    Audits a graph by printing random edges.
    """
    edges = list(graph.edges(data=True))
    if not edges:
        print("No edges found in the graph.")
        return
    
    sample_size = min(n, len(edges))
    audit_edges = random.sample(edges, sample_size)
    
    print(f"\n=== Graph Audit (showing {sample_size} edges) ===")
    clauses = sample.get("clauses", {})
    for src, dst, data in audit_edges:
        print(f"Edge: {src} --[{data['type']}]--> {dst}")
        print(f"  Anchor: '{data['anchor_text']}'")
        src_text = clauses.get(src, "")[:100] + "..."
        dst_text = clauses.get(dst, "")[:100] + "..."
        print(f"  Source: {src_text}")
        print(f"  Dest:   {dst_text}")
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    from data.ace_loader import load_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sample-limit", type=int, default=0,
                        help="0 = 全量（文档要求 graph.json 覆盖整个 train）")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    data_path = Path(args.data_dir)

    samples = json.loads((raw_path / "train.json").read_text(encoding="utf-8"))["scenarios"]
    if args.sample_limit:
        samples = samples[:args.sample_limit]

    output_file = data_path / "graph.json"
    build_all_graphs(samples, output_file)

    # Audit the first sample's graph
    if samples:
        g_data = build_graph(samples[0])
        audit_graph(g_data, samples[0], n=5)
