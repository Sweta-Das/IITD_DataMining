import sys
from graph_utils import read_graphs, get_label, get_edge

def _signature(G):
    # Stronger-than-before signature to deduplicate while preserving ordering.
    # (Not perfect GI-canonical, but much lower collision risk than only label multisets.)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    node_ld = sorted((get_label(G, v), G.degree(v)) for v in G.nodes())
    edge_types = []
    for u, v in G.edges():
        lu, lv = get_label(G, u), get_label(G, v)
        el = get_edge(G, u, v)
        a, b = (lu, lv) if lu <= lv else (lv, lu)
        edge_types.append((a, el, b))
    edge_types.sort()
    return (n, m, tuple(node_ld), tuple(edge_types))

def main():
    db_file = sys.argv[1]
    out_file = sys.argv[2]

    graphs = read_graphs(db_file)

    # PDF: Deduplicate while preserving ordering
    seen = set()
    unique_graphs = []
    for G in graphs:
        sig = _signature(G)
        if sig in seen:
            continue
        seen.add(sig)
        unique_graphs.append(G)

    # Discover label universe from (deduped) DB
    node_labels = set()
    edge_type_keys = set()
    for G in unique_graphs:
        for v in G.nodes():
            node_labels.add(get_label(G, v))
        for u, v in G.edges():
            lu, lv = get_label(G, u), get_label(G, v)
            el = get_edge(G, u, v)
            a, b = (lu, lv) if lu <= lv else (lv, lu)
            edge_type_keys.add(f"{a}-{el}-{b}")

    # Thresholds must match graph_utils.get_strong_features()
    nv_thresholds = [1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100]
    ne_thresholds = [0,1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100,150,200]
    atom_thresholds = list(range(1, 11)) + [12, 15, 20, 30, 40, 50]
    edge_thresholds = list(range(1, 9)) + [10, 12, 15, 20, 30, 40, 50]
    deg_thresholds = [1,2,3,4,5,6,8,10]

    # Hash bucket sizes (must match graph_utils defaults)
    D2 = 1024   # H2
    D3 = 2048   # H3
    DS = 1024   # HS

    feats = []

    # 0) Size lower bounds
    feats += [f"NV>={k}" for k in nv_thresholds]
    feats += [f"NE>={k}" for k in ne_thresholds]

    # 1) Atom label thresholds
    for l in sorted(node_labels):
        feats += [f"A:{l}>={k}" for k in atom_thresholds]

    # 2) Edge type thresholds
    for key in sorted(edge_type_keys):
        feats += [f"E:{key}>={k}" for k in edge_thresholds]

    # 3) Labeled degree thresholds
    for l in sorted(node_labels):
        feats += [f"D:{l}>={k}" for k in deg_thresholds]

    # 4) Cycle presence
    feats.append("CY:any")

    # 5) Hashed wedge/path/star buckets
    feats += [f"H2:{i}" for i in range(D2)]
    feats += [f"H3:{i}" for i in range(D3)]
    feats += [f"HS:{i}" for i in range(DS)]

    with open(out_file, 'w') as f:
        for feat in feats:
            f.write(feat + "\n")

if __name__ == "__main__":
    main()
