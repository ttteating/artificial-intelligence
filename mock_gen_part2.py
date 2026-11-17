def generate_graph(nodes=15):
    nodes_list = []
    for i in range(nodes):
        nodes_list.append({
            "nid": f"n{i:03d}",
            "clause": f"clause_{random.randint(1, 20):03d}",
            "role": random.choice(roles),
            "text": f"条款 {i+1} 描述",
            "sent_idx": random.randint(1, 100)
        })
    edges = []
    for _ in range(random.randint(10, 25)):
        edges.append({
            "src": random.choice([n["nid"] for n in nodes_list]),
            "etype": random.choice(["refers_to", "contradicts", "supports"]),
            "dst": random.choice([n["nid"] for n in nodes_list]),
            "w": round(random.uniform(0.5, 2.0), 2)
        })
    sid = f"tr{random.randint(1000, 9999):04d}"
    return {
        "sid": sid,
        "graph_version": "v2.1",
        "n_nodes": len(nodes_list),
        "n_edges": len(edges),
        "entry": [f"n{random.randint(0, nodes-1):03d}" for _ in range(2)],
        "nodes": nodes_list,
        "edges": edges
    }