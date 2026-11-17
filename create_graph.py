import json, os, random
os.chdir('d:\\program\\artificial intelligence')

roles = ['drafter', 'reviewer', 'approver', 'signatory', 'obligor', 'obligee']

def gen_graph(nodes=15):
    nodes_list = []
    for i in range(nodes):
        nodes_list.append({
            "nid": "n" + str(i).zfill(3),
            "clause": "clause_" + str(random.randint(1, 20)).zfill(3),
            "role": random.choice(roles),
            "text": "条款 " + str(i+1) + " 描述",
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
    sid = "tr" + str(random.randint(1000, 9999)).zfill(4)
    return {
        "sid": sid,
        "graph_version": "v2.1",
        "n_nodes": len(nodes_list),
        "n_edges": len(edges),
        "entry": ["n" + str(random.randint(0, nodes-1)).zfill(3) for _ in range(2)],
        "nodes": nodes_list,
        "edges": edges
    }

graph = gen_graph(15)
with open("mock_graph.json", "w", encoding="utf-8") as f:
    json.dump(graph, f, ensure_ascii=False, indent=2)
print("mock_graph.json generated, sid=" + graph["sid"])