#!/usr/bin/env python3
import sys

def read_original_dataset(filepath):
    graphs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#'):
            graph_id = line[1:]
            i += 1
            
            num_nodes = int(lines[i].strip())
            i += 1
            
            node_labels = []
            for _ in range(num_nodes):
                node_labels.append(lines[i].strip())
                i += 1
      
            num_edges = int(lines[i].strip())
            i += 1

            edges = []
            for _ in range(num_edges):
                edge_parts = lines[i].strip().split()
                if len(edge_parts) == 3:
                    src, dst, label = edge_parts
                    edges.append((int(src), int(dst), label))
                i += 1
            
            graphs.append({
                'id': graph_id,
                'num_nodes': num_nodes,
                'node_labels': node_labels,
                'edges': edges
            })
        else:
            i += 1
    
    return graphs

def convert_to_gspan_gaston(graphs, output_path):

    LABEL_MAPPING = {
        'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4, 'I': 5,
        'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
    }
    
    with open(output_path, 'w') as f:
        for idx, graph in enumerate(graphs):
            f.write(f"t # {idx}\n")
            
           
            for node_id, label in enumerate(graph['node_labels']):
               
                int_label = LABEL_MAPPING.get(label, 99)
                f.write(f"v {node_id} {int_label}\n")
            
            unique_edge_labels = sorted(set([e[2] for e in graph['edges']]))
            edge_label_to_int = {label: i for i, label in enumerate(unique_edge_labels)}
            
            for src, dst, label in graph['edges']:
                f.write(f"e {src} {dst} {edge_label_to_int[label]}\n")

def convert_to_fsg(graphs, output_path):

    with open(output_path, 'w') as f:
        for idx, graph in enumerate(graphs):
            f.write(f"t # {idx}\n")
            
            
            for node_id, label in enumerate(graph['node_labels']):
                f.write(f"v {node_id} {label}\n")
            
           
            for src, dst, label in graph['edges']:
                f.write(f"u {src} {dst} {label}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_dataset.py <input_file> <output_gspan> <output_fsg>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_gspan = sys.argv[2]
    output_fsg = sys.argv[3]
    
    print(f"Reading dataset from {input_file}...")
    graphs = read_original_dataset(input_file)
    print(f"Found {len(graphs)} graphs")
    
    print(f"Converting to gSpan/Gaston format...")
    convert_to_gspan_gaston(graphs, output_gspan)
    print(f"Saved to {output_gspan}")
    
    print(f"Converting to FSG format...")
    convert_to_fsg(graphs, output_fsg)
    print(f"Saved to {output_fsg}")
