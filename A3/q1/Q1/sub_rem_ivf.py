import os
import sys
import time

import numpy as np


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
    - Large / tight budget      : IndexHNSWFlat M=16.
      HNSW has no training phase, so it is the better fit for the 20 s D2
      budget. efConstruction/efSearch are reduced as the budget gets tighter.
    """
    import faiss

    start  = time.perf_counter()
    N, d   = base_vectors.shape
    Q      = len(query_vectors)
    budget = max(0.0, float(time_budget))
    debug_timing = os.environ.get("Q1_TIMING", "0") == "1"

    def t():
        return time.perf_counter() - start

    def log_stage(name: str, seconds: float) -> None:
        if debug_timing:
            print(f"[timing] {name}: {seconds:.6f}s", file=sys.stderr)

    # ── Pre-process ────────────────────────────────────────────────────────
    p_time = time.perf_counter()
    base    = np.ascontiguousarray(base_vectors,  dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    log_stage("preprocess", time.perf_counter() - p_time)

    # ── Index selection ────────────────────────────────────────────────────
    if N <= 80_000:
        # ── Exact search ───────────────────────────────────────────────────
        p_time = time.perf_counter()
        index = faiss.IndexFlatL2(d)
        index.add(base)
        log_stage("index_build", time.perf_counter() - p_time)
    else:
        # ── HNSW ──────────────────────────────────────────────────────────
        # No training phase, so this is a better fit for the larger datasets
        # under a tight wall-clock limit.
        p_time = time.perf_counter()
        index = faiss.IndexHNSWFlat(d, 16)

        if budget >= 60:
            index.hnsw.efConstruction = 100
        elif budget >= 30:
            index.hnsw.efConstruction = 80
        else:
            index.hnsw.efConstruction = 40

        index.add(base)
        log_stage("index_build", time.perf_counter() - p_time)

        rem = budget - t()
        if rem > 40:
            index.hnsw.efSearch = 128
        elif rem > 20:
            index.hnsw.efSearch = 64
        else:
            index.hnsw.efSearch = 32

    # ── ANN search ─────────────────────────────────────────────────────────
    p_search = time.perf_counter()
    _dist, I = index.search(queries, k)   # I: (Q, k) – indices of k-NNs
    log_stage("final_search", time.perf_counter() - p_search)

    # ── Frequency counting ─────────────────────────────────────────────────
    p_count = time.perf_counter()
    flat = I.ravel()
    flat = flat[flat >= 0]
    counts = np.bincount(flat, minlength=N)
    log_stage("counting", time.perf_counter() - p_count)

    # ── Rank: descending count, ties broken by ascending index ─────────────
    # np.argsort(kind='stable') preserves original order (0, 1, 2, …) for
    # equal counts, giving the required lower-index-first tie-breaking.
    p_rank = time.perf_counter()
    ranking = np.argsort(-counts, kind='stable')
    log_stage("ranking", time.perf_counter() - p_rank)
    log_stage("total_solve", t())

    return ranking[:K].astype(np.int64)
