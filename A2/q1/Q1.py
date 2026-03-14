import sys
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data(arg):
    """Load dataset from either a .npy file or a URL based on the argument."""

    if arg.endswith('.npy'):
        try:
            return np.load(arg)
        except Exception as e:
            print(f"Error loading .npy file: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        try:
            dataset_num = int(arg)
            url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id=aib252564&dataset_num={dataset_num}"
            with urllib.request.urlopen(url) as response:
                raw_data = response.read().decode('utf-8')
                data = json.loads(raw_data)

                with open(f"dataset_{dataset_num}.npy", 'wb') as f:
                    np.save(f, np.array(data["X"]))
                return np.array(data["X"])
            
        except ValueError:
            print("Argument must be either dataset_num (int) or path to .npy file", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error fetching dataset from URL: {e}", file=sys.stderr)
            sys.exit(1)


def find_elbow_point(k_values, inertias):
    points = np.column_stack((k_values, inertias))
    line_start = points[0]
    line_end = points[-1]
    line_vec = line_end - line_start
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    max_dist = -1
    elbow_k = -1
    
    for i in range(len(points)):
        point = points[i]
        point_vec = point - line_start
        proj = np.dot(point_vec, line_vec_norm) * line_vec_norm
        perp_vec = point_vec - proj
        dist = np.linalg.norm(perp_vec)
        
        if dist > max_dist:
            max_dist = dist
            elbow_k = k_values[i]
            
    return elbow_k


def main():
    if len(sys.argv) != 2:
        print("Usage:", file=sys.stderr)
        print("  python3 Q1.py <dataset_num>", file=sys.stderr)
        print("  python3 Q1.py <path_to_dataset>.npy", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    X = load_data(arg)

    if len(X) == 0:
        print("Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    inertias = []

    max_k = min(15, len(X))
    k_range = list(range(1, max_k + 1))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
    optimal_k = find_elbow_point(k_range, inertias)

    plt.figure()
    plt.plot(k_range, inertias, 'bo-')
    plt.plot(optimal_k, inertias[k_range.index(optimal_k)], 'ro', markersize=10, label=f'Optimal k={optimal_k}')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Objective Value (Inertia)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot.png')
    
    print(optimal_k)

if __name__ == "__main__":
    main()
