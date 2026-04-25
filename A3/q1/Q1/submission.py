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
    - Safe HNSW window          : IndexHNSWFlat with conservative build
      parameters, used only when the estimated build cost is comfortably
      below the time budget.
    - Otherwise                 : lightweight IndexIVFFlat fallback with a
      smaller coarse quantizer and no probe calibration.
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

    def log_value(name: str, value: float) -> None:
        if debug_timing:
            print(f"[info] {name}: {value}", file=sys.stderr)

    # ── Pre-process ────────────────────────────────────────────────────────
    p_time = time.perf_counter()
    base    = np.ascontiguousarray(base_vectors,  dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    log_stage("preprocess", time.perf_counter() - p_time)
    log_value("N_base", float(N))
    log_value("Q_queries", float(Q))

    # ── Index selection ────────────────────────────────────────────────────
    if N <= 80_000:
        # ── Exact search ───────────────────────────────────────────────────
        p_time = time.perf_counter()
        index = faiss.IndexFlatL2(d)
        index.add(base)
        log_stage("index_build", time.perf_counter() - p_time)
    else:
        # Conservative estimate for HNSW build time. The earlier pure-HNSW
        # attempt exceeded the D2 budget, so we only use HNSW when the
        # expected build cost is safely below the wall-clock limit.
        hnsw_est_secs = N * 50e-6

        if hnsw_est_secs < 0.45 * budget:
            # ── HNSW ──────────────────────────────────────────────────────
            p_time = time.perf_counter()
            index = faiss.IndexHNSWFlat(d, 12)

            if budget >= 60:
                index.hnsw.efConstruction = 96
            elif budget >= 30:
                index.hnsw.efConstruction = 64
            else:
                index.hnsw.efConstruction = 48

            index.add(base)
            log_stage("index_build", time.perf_counter() - p_time)

            rem = budget - t()
            if rem > 40:
                index.hnsw.efSearch = 96
            elif rem > 20:
                index.hnsw.efSearch = 64
            else:
                index.hnsw.efSearch = 32

        else:
            # ── Light IVF fallback ───────────────────────────────────────
            p_time = time.perf_counter()
            nlist = int(np.clip(np.sqrt(N), 64, 1024))
            train_size = min(N, 40 * nlist)
            rng = np.random.default_rng(42)

            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

            p_train = time.perf_counter()
            index.train(base[rng.choice(N, train_size, replace=False)])
            log_stage("index_train", time.perf_counter() - p_train)

            p_add = time.perf_counter()
            index.add(base)
            log_stage("index_add", time.perf_counter() - p_add)
            log_stage("index_build", time.perf_counter() - p_time)

            # Pilot-based probe selection keeps the fallback adaptive across
            # different hidden datasets and runtime environments.
            pilot_nprobe = 8
            pilot_queries = min(Q, 50)
            safety_margin = 0.75

            index.nprobe = pilot_nprobe
            p_probe = time.perf_counter()
            index.search(queries[:pilot_queries], k)
            pilot_seconds = time.perf_counter() - p_probe
            log_stage("probe_calibration", pilot_seconds)

            t_per_nprobe = (pilot_seconds / pilot_nprobe) * (Q / pilot_queries)
            rem = budget - t() - safety_margin
            nprobe = int(np.clip(rem / max(t_per_nprobe, 1e-3), 8, 32))
            index.nprobe = min(nprobe, nlist)
            log_stage("nprobe_selected", float(index.nprobe))

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
