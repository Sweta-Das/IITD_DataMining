import networkx as nx
from collections import Counter

def read_graphs(file_path):
    """
    Robust graph reader handling 't # id' and '# id' formats.
    """
    graphs = []
    current_graph = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if parts[0] == 't' or parts[0] == '#':
                if current_graph is not None:
                    graphs.append(current_graph)
                current_graph = nx.Graph()
                if len(parts) > 2 and parts[1] == '#':
                    current_graph.graph['id'] = parts[2]
                elif len(parts) > 1:
                    current_graph.graph['id'] = parts[1]
                else:
                    current_graph.graph['id'] = len(graphs)
            elif parts[0] == 'v':
                if current_graph is None:
                    current_graph = nx.Graph()
                current_graph.add_node(int(parts[1]), label=parts[2])
            elif parts[0] == 'e':
                if current_graph is None:
                    current_graph = nx.Graph()
                current_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])

    if current_graph is not None:
        graphs.append(current_graph)
    return graphs

def get_label(G, n):
    return G.nodes[n]['label']

def get_edge(G, u, v):
    return G.edges[u, v]['label']

def _md5_bucket(s: str, D: int) -> int:
    # Stable across machines/runs
    import hashlib
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % D

def get_strong_features(G, D2=1024, D3=2048, DS=1024):
    """
    Safe (no-false-negative) feature set with stronger structural signals.

    - Count thresholds for |V|, |E|, atom labels, edge types, labeled degrees
    - Hashed edge-aware wedges (length-2) with count thresholds  (H2:*)
    - Hashed edge-labeled length-3 paths with count thresholds       (H3:*)
    - Hashed edge-aware stars (local neighbor-type -> global count)  (HS:*)
    - Cycle presence (CY:any)

    All features are monotone: if q is a subgraph of g, then every feature that is 1 in q is also 1 in g.
    """
    feats = set()

    # ----------------------------
    # 0) Graph size lower bounds
    # ----------------------------
    nv = G.number_of_nodes()
    ne = G.number_of_edges()

    nv_thresholds = [1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100]
    ne_thresholds = [0,1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100,150,200]

    for k in nv_thresholds:
        if nv >= k:
            feats.add(f"NV>={k}")
    for k in ne_thresholds:
        if ne >= k:
            feats.add(f"NE>={k}")

    # ----------------------------
    # 1) Atom label counts
    # ----------------------------
    atom_thresholds = list(range(1, 11)) + [12, 15, 20, 30, 40, 50]
    atoms = Counter(get_label(G, n) for n in G.nodes())
    for l, c in atoms.items():
        for k in atom_thresholds:
            if c >= k:
                feats.add(f"A:{l}>={k}")

    # ----------------------------
    # 2) Edge type counts (endpoint labels + edge label)
    # ----------------------------
    edge_thresholds = list(range(1, 9)) + [10, 12, 15, 20, 30, 40, 50]
    edges = Counter()
    for u, v in G.edges():
        l1, l2 = sorted([get_label(G, u), get_label(G, v)])
        el = get_edge(G, u, v)
        edges[f"{l1}-{el}-{l2}"] += 1

    for key, c in edges.items():
        for k in edge_thresholds:
            if c >= k:
                feats.add(f"E:{key}>={k}")

    # ----------------------------
    # 3) Labeled degree thresholds
    # ----------------------------
    deg_thresholds = [1,2,3,4,5,6,8,10]
    for n in G.nodes():
        l = get_label(G, n)
        d = G.degree(n)
        for k in deg_thresholds:
            if d >= k:
                feats.add(f"D:{l}>={k}")

    # ----------------------------
    # 4) Cycle presence (monotone)
    # ----------------------------
    try:
        if nx.cycle_basis(G):
            feats.add("CY:any")
    except Exception:
        # In case of any unexpected NetworkX issue, just skip cycle feature (never creates FN).
        pass

    # ----------------------------
    # 5) Hashed edge-aware wedges (length-2)
    # ----------------------------
    # Pattern = (nbr1_label, edge1_label) -- center_label -- (edge2_label, nbr2_label)
    # Canonicalize by sorting the two arms.
    wedge_counts = Counter()
    for c in G.nodes():
        nbrs = list(G.neighbors(c))
        if len(nbrs) < 2:
            continue
        cl = get_label(G, c)
        # Build arm descriptors
        arms = []
        for x in nbrs:
            el = get_edge(G, c, x)
            xl = get_label(G, x)
            arms.append((el, xl))
        # Count unordered pairs of arms
        for i in range(len(arms)):
            for j in range(i+1, len(arms)):
                a1, a2 = arms[i], arms[j]
                if a2 < a1:
                    a1, a2 = a2, a1
                key = f"{cl}|{a1[0]}:{a1[1]}|{a2[0]}:{a2[1]}"
                wedge_counts[key] += 1

    wedge_count_thresholds = [1,2,3,5]
    for key, c in wedge_counts.items():
        for t in wedge_count_thresholds:
            if c >= t:
                idx = _md5_bucket("H2|" + key + f"|t={t}", D2)
                feats.add(f"H2:{idx}")

    # ----------------------------
    # 6) Hashed edge-labeled simple paths of length 3 (4 nodes, 3 edges)
    # ----------------------------
    # Signature includes node labels and edge labels. Canonicalize by choosing
    # lexicographically smaller between forward and reversed sequences.
    path3_counts = Counter()

    # DFS from each start node, record only if start_id < end_id to avoid duplicates
    for s in G.nodes():
        stack = [(s, [s], [])]  # (current, node_path, edge_label_path)
        while stack:
            cur, node_path, edge_path = stack.pop()
            depth = len(edge_path)
            if depth == 3:
                t = cur
                if s < t:
                    nlabels = [get_label(G, x) for x in node_path]
                    # build forward string
                    fwd = f"{nlabels[0]}-{edge_path[0]}-{nlabels[1]}-{edge_path[1]}-{nlabels[2]}-{edge_path[2]}-{nlabels[3]}"
                    # build reverse string
                    rev = f"{nlabels[3]}-{edge_path[2]}-{nlabels[2]}-{edge_path[1]}-{nlabels[1]}-{edge_path[0]}-{nlabels[0]}"
                    key = fwd if fwd <= rev else rev
                    path3_counts[key] += 1
                continue

            for nb in G.neighbors(cur):
                if nb in node_path:
                    continue
                el = get_edge(G, cur, nb)
                stack.append((nb, node_path + [nb], edge_path + [el]))

    path3_count_thresholds = [1,2,3,5]
    for key, c in path3_counts.items():
        for t in path3_count_thresholds:
            if c >= t:
                idx = _md5_bucket("H3|" + key + f"|t={t}", D3)
                feats.add(f"H3:{idx}")

    # ----------------------------
    # 7) Hashed edge-aware stars (local neighbor-type counts -> global count)
    # ----------------------------
    # For each node: count neighbors of type (edge_label, neighbor_label).
    # Local thresholds (>=1,>=2,>=3) define a "property" of that node.
    # Then count how many nodes satisfy that property in the graph, and apply
    # global thresholds (>=1,>=2,>=3,>=5). Hash into DS buckets.
    local_props = Counter()
    local_thr = [1,2,3]
    global_thr = [1,2,3,5]

    for c in G.nodes():
        if G.degree(c) == 0:
            continue
        cl = get_label(G, c)
        nbr_type_counts = Counter()
        for nb in G.neighbors(c):
            et = get_edge(G, c, nb)
            nl = get_label(G, nb)
            nbr_type_counts[(et, nl)] += 1

        for (et, nl), cnt in nbr_type_counts.items():
            for t in local_thr:
                if cnt >= t:
                    prop = f"{cl}|{et}:{nl}|local>={t}"
                    local_props[prop] += 1

    for prop, cnt in local_props.items():
        for gt in global_thr:
            if cnt >= gt:
                idx = _md5_bucket("HS|" + prop + f"|g>={gt}", DS)
                feats.add(f"HS:{idx}")

    return feats
