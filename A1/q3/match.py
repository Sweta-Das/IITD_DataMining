import sys
import numpy as np

def main():
    db_path = sys.argv[1]
    query_path = sys.argv[2]
    out_path = sys.argv[3]

    # Load binary feature matrices
    db_matrix = np.load(db_path)    # (N_db, F), uint8
    q_matrix  = np.load(query_path) # (N_q, F), uint8

    with open(out_path, 'w') as f:
        for q_idx in range(q_matrix.shape[0]):
            q_vec = q_matrix[q_idx]

            # Candidate if every 1-bit in q is also 1 in db:
            # db >= q elementwise for binary vectors.
            matches = np.all(db_matrix >= q_vec, axis=1)
            cand = np.where(matches)[0] + 1  # 1-based serial numbers

            # Output format required by spec:
            # q # <query_serial>
            # c # <cand_serials...>
            f.write(f"q # {q_idx + 1}\n")
            if cand.size == 0:
                f.write("c #\n")
            else:
                # Preserve original DB ordering (already increasing by construction)
                f.write("c # " + " ".join(map(str, cand.tolist())) + "\n")

if __name__ == "__main__":
    main()
