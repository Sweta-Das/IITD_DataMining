import numpy as np
import time


def solve(base_vectors, query_vectors, k, K, time_budget):
    """
    Returns the K most representative database items.

    Algorithm
    ---------
    1. Build a FAISS index on base_vectors.
    2. For every query retrieve its k nearest neighbours.
    3. Count how often each base item appears across all retrieved sets.
    4. Return the top-K items sorted by count (desc), ties by index (asc).

    Index selection (adaptive to N and time_budget)
    ------------------------------------------------
    - Tiny  (N ≤ 80 000)       : exact IndexFlatL2 – always fast enough.
    - Small/medium (HNSW fits)  : IndexHNSWFlat M=16.
      HNSW builds in ~40 µs/vector. We use it when the estimated build time
      < 35 % of the budget so there is plenty of time for queries.
      efSearch is calibrated on remaining time after build.
    - Large / tight budget      : IndexIVFFlat.
      nlist ≈ 4√N (clamped to [64, 8192]).
      Training uses 20 × nlist samples (minimal valid set) for tight budgets
      so more wall-time remains for the actual search phase.
      nprobe is calibrated by timing 2 000 probe queries at nprobe=8,
      extrapolating to the full query set, and fitting as many probes
      as possible in the remaining time minus a 2 s safety margin.
    """
    import faiss

    start  = time.perf_counter()
    N, d   = base_vectors.shape
    Q      = len(query_vectors)
    budget = max(0.0, float(time_budget))

    def t():
        return time.perf_counter() - start

    # ── Pre-process ────────────────────────────────────────────────────────
    base    = np.ascontiguousarray(base_vectors,  dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # ── Index selection ────────────────────────────────────────────────────
    HNSW_BUILD_US   = 40e-6      # conservative: 40 µs / vector for M=16
    hnsw_est_secs   = N * HNSW_BUILD_US

    if N <= 80_000:
        # ── Exact search ───────────────────────────────────────────────────
        index = faiss.IndexFlatL2(d)
        index.add(base)

    elif hnsw_est_secs < 0.35 * budget:
        # ── HNSW ──────────────────────────────────────────────────────────
        # High-recall graph index; fast queries after one-time build.
        index = faiss.IndexHNSWFlat(d, 16)
        index.hnsw.efConstruction = 100
        index.add(base)                        # graph built here

        rem = budget - t()
        if   rem > 40: index.hnsw.efSearch = 256
        elif rem > 20: index.hnsw.efSearch = 128
        elif rem > 10: index.hnsw.efSearch = 64
        else:          index.hnsw.efSearch = 32

    else:
        # ── IVFFlat with calibrated nprobe ─────────────────────────────────
        nlist = int(np.clip(4 * np.sqrt(N), 64, 8192))

        # Use fewer training samples for tight budgets so more time is
        # left for the actual ANN search (each extra nprobe matters a lot).
        if   budget >= 60: train_factor = 50
        elif budget >= 30: train_factor = 30
        else:              train_factor = 20   # minimum viable; FAISS warns
                                               # but empirically gives better
                                               # nDCG than spending 6 s extra
                                               # on training.

        train_size = min(N, train_factor * nlist)
        rng = np.random.default_rng(42)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(base[rng.choice(N, train_size, replace=False)])
        index.add(base)

        # ── Calibrate nprobe ──────────────────────────────────────────────
        # Time 2000 queries at nprobe=8, then extrapolate to full Q.
        probe_nprobe = 8
        probe_size   = min(Q, 2000)

        index.nprobe = probe_nprobe
        t0 = time.perf_counter()
        index.search(queries[:probe_size], k)
        t_per_np = (time.perf_counter() - t0) / probe_nprobe * (Q / probe_size)
        # t_per_np = estimated seconds per one nprobe unit for the full Q

        safety   = 2.0                # reserve 2 s to avoid wall-clock overshoot
        rem      = budget - t() - safety
        if rem <= 0.0:
            nprobe = 1
        else:
            nprobe = int(np.clip(rem / max(t_per_np, 1e-3), 1, nlist))
        index.nprobe = nprobe

    # ── ANN search ─────────────────────────────────────────────────────────
    _dist, I = index.search(queries, k)   # I: (Q, k) – indices of k-NNs

    # ── Frequency counting ─────────────────────────────────────────────────
    flat = I.ravel()
    flat = flat[flat >= 0]
    counts = np.bincount(flat, minlength=N)

    # ── Rank: descending count, ties broken by ascending index ─────────────
    # np.argsort(kind='stable') preserves original order (0, 1, 2, …) for
    # equal counts, giving the required lower-index-first tie-breaking.
    ranking = np.argsort(-counts, kind='stable')

    return ranking[:K].astype(np.int64)
