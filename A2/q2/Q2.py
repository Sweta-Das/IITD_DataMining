import sys
import time
import heapq
import random
from collections import defaultdict

def main():
    if len(sys.argv) < 7:
        print("Usage: python3 Q2.py <graph> <seed> <out> <k> <r> <hops>")
        sys.exit(1)
        
    graph_file = sys.argv[1]
    seed_file = sys.argv[2]
    out_file = sys.argv[3]
    k = int(sys.argv[4])
    r = int(sys.argv[5])
    hops = int(sys.argv[6])
    
    A0 = []
    with open(seed_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            A0.extend([int(x) for x in parts])
    A0 = list(set(A0))
    
    adj = defaultdict(list)
    edges_list = []
    
    with open(graph_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) >= 3:
                u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            else:
                u, v = int(parts[0]), int(parts[1])
                p = 1.0
                
            mask = 0
            for i in range(r):
                if random.random() < p:
                    mask |= (1 << i)
            if mask > 0:
                adj[u].append((v, mask))
                edges_list.append((u, v, mask))

    adj = {u: adj[u] for u in adj}
    
    blocked_edges = set()
    output_lines = []
    
    def write_output():
        out = list(output_lines)
        e_idx = 0
        while len(out) < k and e_idx < len(edges_list):
            e = edges_list[e_idx]
            tupe = (e[0], e[1])
            if tupe not in blocked_edges:
                out.append(tupe)
            e_idx += 1
        while len(out) < k:
            out.append((0, 0))
            
        with open(out_file, 'w') as f:
            for u, v in out:
                f.write(f"{u} {v}\n")
                
    write_output()
    
    if hops != -1:
        def get_h_hop_reachability(blocked, return_edges=False):
            total_reach = 0
            candidate_edges = set()
            for i in range(r):
                visited = set(A0)
                q = A0[:]
                depths = {u: 0 for u in A0}
                q_idx = 0
                while q_idx < len(q):
                    u = q[q_idx]
                    q_idx += 1
                    d = depths[u]
                    if d == hops:
                        continue
                    if u in adj:
                        for v, mask in adj[u]:
                            if (mask & (1 << i)):
                                if (u, v) not in blocked:
                                    if return_edges:
                                        candidate_edges.add((u, v))
                                    if v not in visited:
                                        visited.add(v)
                                        depths[v] = d + 1
                                        q.append(v)
                total_reach += len(visited)
            return total_reach, candidate_edges

        current_reach, candidates = get_h_hop_reachability(blocked_edges, return_edges=True)
        
        celf_queue = []
        for e in candidates:
            blocked_edges.add(e)
            new_reach, _ = get_h_hop_reachability(blocked_edges)
            blocked_edges.remove(e)
            gain = current_reach - new_reach
            if gain > 0:
                celf_queue.append((-gain, e))
        heapq.heapify(celf_queue)
        
        for step in range(k):
            best_edge = None
            while celf_queue:
                neg_gain, e = heapq.heappop(celf_queue)
                
                blocked_edges.add(e)
                new_r, _ = get_h_hop_reachability(blocked_edges)
                blocked_edges.remove(e)
                
                actual_gain = current_reach - new_r
                if actual_gain <= 0:
                    continue
                    
                if not celf_queue or actual_gain >= -celf_queue[0][0]:
                    best_edge = e
                    current_reach = new_r
                    break
                else:
                    heapq.heappush(celf_queue, (-actual_gain, e))
            
            if best_edge:
                blocked_edges.add(best_edge)
                output_lines.append(best_edge)
                write_output()
            else:
                break
                
    else:
        super_root = -1
        A0_set = set(A0)
        
        def compute_dominator_gains(blocked):
            marginal_gains = defaultdict(int)
            for i in range(r):
                local_adj = defaultdict(list)
                local_rev = defaultdict(list)
                
                q = A0[:]
                visited = set(A0)
                q_idx = 0
                while q_idx < len(q):
                    u = q[q_idx]
                    q_idx += 1
                    if u in adj:
                        for v, mask in adj[u]:
                            if (mask & (1 << i)) and (u, v) not in blocked:
                                local_adj[u].append(v)
                                local_rev[v].append(u)
                                if v not in visited:
                                    visited.add(v)
                                    q.append(v)
                
                for a in A0:
                    local_rev[a] = [super_root]
                local_adj[super_root] = list(A0)
                
                rpo = []
                dfs_visited = set([super_root])
                stack = [(super_root, list(local_adj[super_root]))]
                while stack:
                    u, neighbors = stack[-1]
                    if neighbors:
                        v = neighbors.pop()
                        if v not in dfs_visited:
                            dfs_visited.add(v)
                            stack.append((v, list(local_adj[v])))
                    else:
                        stack.pop()
                        rpo.append(u)
                rpo.reverse()
                
                rpo_idx = {u: idx for idx, u in enumerate(rpo)}
                doms = {u: None for u in rpo}
                doms[super_root] = super_root
                
                def intersect(b1, b2):
                    finger1, finger2 = b1, b2
                    while finger1 != finger2:
                        while rpo_idx[finger1] > rpo_idx[finger2]:
                            finger1 = doms[finger1]
                        while rpo_idx[finger2] > rpo_idx[finger1]:
                            finger2 = doms[finger2]
                    return finger1

                changed = True
                while changed:
                    changed = False
                    for u in rpo:
                        if u == super_root: continue
                        new_idom = None
                        for v in local_rev[u]:
                            if v in doms and doms[v] is not None:
                                if new_idom is None:
                                    new_idom = v
                                else:
                                    new_idom = intersect(v, new_idom)
                        if doms[u] != new_idom:
                            doms[u] = new_idom
                            changed = True
                
                dom_children = defaultdict(list)
                for u in rpo:
                    if u != super_root and doms[u] is not None:
                        dom_children[doms[u]].append(u)
                        
                subtree_size = {}
                dfs_in = {}
                dfs_out = {}
                timer = 0
                
                stack_tree = [(super_root, False)]
                while stack_tree:
                    u, is_post = stack_tree.pop()
                    if is_post:
                        size = (1 if u != super_root and u not in A0_set else 0)
                        for c in dom_children[u]:
                            size += subtree_size[c]
                        subtree_size[u] = size
                        timer += 1
                        dfs_out[u] = timer
                    else:
                        timer += 1
                        dfs_in[u] = timer
                        stack_tree.append((u, True))
                        for c in reversed(dom_children[u]):
                            stack_tree.append((c, False))
                
                for v in rpo:
                    if v == super_root or v in A0_set:
                        continue
                    idom_v = doms[v]
                    if idom_v == super_root:
                        continue
                        
                    is_in_neighbor = False
                    for w in local_rev[v]:
                        if w == idom_v:
                            is_in_neighbor = True
                            break
                    if not is_in_neighbor:
                        continue
                        
                    is_bridge = True
                    for w in local_rev[v]:
                        if w == idom_v: continue
                        if w not in dfs_in: 
                            is_bridge = False
                            break
                        if not (dfs_in[v] <= dfs_in[w] and dfs_out[w] <= dfs_out[v]):
                            is_bridge = False
                            break
                    
                    if is_bridge:
                        marginal_gains[(idom_v, v)] += subtree_size[v]
                        
            return marginal_gains

        for step in range(k):
            gains = compute_dominator_gains(blocked_edges)
            if not gains:
                break
            
            best_edge = max(gains.items(), key=lambda x: x[1])[0]
            blocked_edges.add(best_edge)
            output_lines.append(best_edge)
            write_output()

if __name__ == "__main__":
    main()
