#!/usr/bin/env python3
"""
Simple indexing test for JIST descriptors.

Given database descriptors and query descriptors, this script builds a FAISS index
and runs approximate/flat search to return top-k matches. Optional label files
can be provided to compute recall@1/5/10 based on label equality.
"""

import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import faiss


def setup_logging(output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file = output_folder / "log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging to {log_file}")


def load_descriptors(path: Path) -> np.ndarray:
    desc = np.load(path)
    if desc.ndim != 2:
        raise ValueError(f"Descriptors at {path} must be 2D (num_vectors x dim), got shape {desc.shape}")
    return desc.astype(np.float32)


def normalize_features(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_index(db_desc: np.ndarray, metric: str):
    dim = db_desc.shape[1]
    if metric == "cosine":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(db_desc)
    return index


def compute_recalls(pred_indices: np.ndarray, query_labels: np.ndarray, db_labels: np.ndarray):
    recalls_at = [1, 5, 10]
    recalls = []
    for k in recalls_at:
        hits = 0
        for i, preds in enumerate(pred_indices[:, :k]):
            if np.any(db_labels[preds] == query_labels[i]):
                hits += 1
        recalls.append(100.0 * hits / len(query_labels))
    return {f"R@{k}": r for k, r in zip(recalls_at, recalls)}


def main():
    parser = argparse.ArgumentParser(
        description="Test FAISS indexing on JIST descriptors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db_desc", required=True, type=Path, help="Path to database descriptors .npy")
    parser.add_argument("--query_desc", required=True, type=Path, help="Path to query descriptors .npy")
    parser.add_argument("--db_labels", type=Path, default=None, help="Optional .npy labels for database vectors")
    parser.add_argument("--query_labels", type=Path, default=None, help="Optional .npy labels for query vectors")
    parser.add_argument("--k", type=int, default=10, help="Top-k to retrieve")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"], help="Similarity metric")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize descriptors before indexing")
    parser.add_argument("--output", type=Path, default=Path("index_test"), help="Output folder for results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.output / timestamp
    setup_logging(out_dir)
    logging.info(f"Arguments: {args}")

    db_desc = load_descriptors(args.db_desc)
    query_desc = load_descriptors(args.query_desc)
    if db_desc.shape[1] != query_desc.shape[1]:
        raise ValueError(f"Dim mismatch: db dim {db_desc.shape[1]} vs query dim {query_desc.shape[1]}")

    if args.normalize or args.metric == "cosine":
        db_desc = normalize_features(db_desc)
        query_desc = normalize_features(query_desc)

    index = build_index(db_desc, args.metric)
    logging.info(f"Built index with {db_desc.shape[0]} vectors, dim={db_desc.shape[1]}, metric={args.metric}")

    logging.info("Searching...")
    scores, indices = index.search(query_desc, args.k)
    logging.info(f"Search done. Results shape: {indices.shape}")

    # Save results
    np.save(out_dir / "topk_indices.npy", indices)
    np.save(out_dir / "topk_scores.npy", scores)
    logging.info(f"Saved indices and scores to {out_dir}")

    summary = {
        "db_vectors": int(db_desc.shape[0]),
        "query_vectors": int(query_desc.shape[0]),
        "dim": int(db_desc.shape[1]),
        "metric": args.metric,
        "normalized": bool(args.normalize or args.metric == "cosine"),
        "k": args.k,
    }

    # Optional recalls
    if args.db_labels and args.query_labels:
        db_labels = np.load(args.db_labels)
        query_labels = np.load(args.query_labels)
        if len(db_labels) != db_desc.shape[0]:
            raise ValueError("db_labels length must match db descriptors")
        if len(query_labels) != query_desc.shape[0]:
            raise ValueError("query_labels length must match query descriptors")
        recalls = compute_recalls(indices, query_labels, db_labels)
        summary.update(recalls)
        logging.info("Recalls: " + ", ".join([f"{k}: {v:.2f}%" for k, v in recalls.items()]))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
