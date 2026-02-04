import sys
import numpy as np
from graph_utils import read_graphs, get_strong_features

def main():
    graph_file = sys.argv[1]
    feature_file = sys.argv[2]
    out_file = sys.argv[3]
    
    # 1. Load Feature Definitions
    with open(feature_file, 'r') as f:
        feats = [line.strip() for line in f]
    feat_map = {feat: i for i, feat in enumerate(feats)}
    num_feats = len(feats)
    
    # 2. Load Graphs
    graphs = read_graphs(graph_file)
    
    # 3. Create Binary Vector Matrix (N x F)
    # Using uint8 to save memory
    matrix = np.zeros((len(graphs), num_feats), dtype=np.uint8)
    
    for i, G in enumerate(graphs):
        graph_feats = get_strong_features(G)
        for gf in graph_feats:
            if gf in feat_map:
                matrix[i, feat_map[gf]] = 1
                
    # 4. Save as .npy
    np.save(out_file, matrix)

if __name__ == "__main__":
    main()