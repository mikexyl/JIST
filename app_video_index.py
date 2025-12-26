#!/usr/bin/env python3
"""
Streamlit app to:
- Preview a video (sampled every N frames)
- Load query/database descriptors
- Run FAISS search
- Visualize top-k results for a selected query

Usage:
  streamlit run app_video_index.py --server.port 8501

Requirements (pip install if missing):
  streamlit opencv-python faiss-cpu pillow numpy
"""

import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import streamlit as st
import faiss
from PIL import Image

# -------------------------------
# Helpers
# -------------------------------

def load_video_frames(video_path: Path, sample_rate: int, max_frames: int | None = 200):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

def load_descriptors(desc_path: Path) -> np.ndarray:
    arr = np.load(desc_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D descriptors, got {arr.shape}")
    return arr.astype(np.float32)

def normalize_features(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def build_index(db_desc: np.ndarray, metric: str):
    dim = db_desc.shape[1]
    if metric == "cosine":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(db_desc)
    return index

def run_search(db_desc: np.ndarray, query_desc: np.ndarray, k: int, metric: str):
    if db_desc.shape[1] != query_desc.shape[1]:
        raise ValueError(f"Dim mismatch: db {db_desc.shape[1]} vs query {query_desc.shape[1]}")
    if metric == "cosine":
        db_desc = normalize_features(db_desc)
        query_desc = normalize_features(query_desc)
    index = build_index(db_desc, metric)
    scores, indices = index.search(query_desc, k)
    return scores, indices


def slice_sequence(frames, seq_idx: int, seq_length: int, stride: int):
    """
    Extract a sequence of frames. If the index is out of bounds,
    wrap around or clamp to available frames.
    """
    if not frames:
        return []
    
    # Calculate start position
    start = seq_idx * stride
    
    # If start is beyond frames, use modulo to wrap around
    # or clamp to show the last available sequence
    if start >= len(frames):
        # Try wrapping around for cyclic view
        start = start % len(frames) if len(frames) > 0 else 0
        # Alternative: clamp to last possible sequence
        # start = max(0, len(frames) - seq_length)
    
    end = min(start + seq_length, len(frames))
    seq = frames[start:end]
    
    # Pad if short (happens at the end of video)
    if len(seq) < seq_length and seq:
        seq = seq + [seq[-1]] * (seq_length - len(seq))
    
    return seq

def safe_load_labels(path: str | None):
    if path is None or path == "":
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")
    return np.load(p)

# -------------------------------
# Streamlit UI
# -------------------------------

def main():
    st.title("JIST Video + Index Viewer")
    st.caption("Preview a video, then search query descriptors against a database and visualize top-k matches.")

    # Persistent container for search results to prevent reset on slider changes
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None

    with st.sidebar:
        st.header("Inputs")
        video_path = st.text_input("Video path (used if specific paths below are empty)")
        query_video_path = st.text_input("Query video path (optional)")
        db_video_path = st.text_input("Database video path (optional)")
        sample_rate = st.number_input("Sample every N frames", min_value=1, max_value=50, value=5, step=1)
        max_preview = st.number_input("Max preview frames", min_value=10, max_value=1500, value=300, step=10)
        query_desc_path = st.text_input("Query descriptors (.npy)")
        db_desc_path = st.text_input("Database descriptors (.npy)")
        db_labels_path = st.text_input("Database labels (.npy, optional)", value="")
        query_labels_path = st.text_input("Query labels (.npy, optional)", value="")
        metric = st.selectbox("Metric", options=["cosine", "l2"], index=0)
        k = st.number_input("Top-k", min_value=1, max_value=50, value=10, step=1)
        seq_length = st.number_input("Sequence length (for visualization)", min_value=1, max_value=50, value=5, step=1)
        stride = st.number_input("Sequence stride (for visualization)", min_value=1, max_value=50, value=1, step=1)
        run_button = st.button("Run search")

    # Preview video
    def load_frames_for(role: str, path_str: str):
        if not path_str:
            return []
        p = Path(path_str)
        if not p.exists():
            st.warning(f"{role} video not found: {p}")
            return []
        try:
            return load_video_frames(p, sample_rate=sample_rate, max_frames=int(max_preview))
        except Exception as e:
            st.error(f"{role} video load error: {e}")
            return []

    q_video_frames = load_frames_for("Query", query_video_path or video_path)
    db_video_frames = load_frames_for("Database", db_video_path or video_path)

    if q_video_frames:
        st.subheader("Query video preview")
        st.write(f"Showing up to {len(q_video_frames)} frames (sampled every {sample_rate} frames)")
        cols = st.columns(4)
        for i, frame in enumerate(q_video_frames[:8]):
            cols[i % 4].image(frame, caption=f"Q Frame {i}", width=300)

    if db_video_frames and db_video_frames is not q_video_frames:
        st.subheader("Database video preview")
        st.write(f"Showing up to {len(db_video_frames)} frames (sampled every {sample_rate} frames)")
        cols = st.columns(4)
        for i, frame in enumerate(db_video_frames[:8]):
            cols[i % 4].image(frame, caption=f"DB Frame {i}", width=300)

    # Run search and persist results on click; reuse on slider changes
    if run_button:
        try:
            q_desc = load_descriptors(Path(query_desc_path))
            db_desc = load_descriptors(Path(db_desc_path))
            scores, indices = run_search(db_desc, q_desc, int(k), metric)

            st.session_state["search_results"] = {
                "scores": scores,
                "indices": indices,
                "q_len": q_desc.shape[0],
                "db_len": db_desc.shape[0],
                "dim": db_desc.shape[1],
                "metric": metric,
                "k": int(k),
            }
            st.success("Search completed (results saved; adjusting the slider won't clear them)")
        except Exception as e:
            st.error(f"Search error: {e}")

    results = st.session_state.get("search_results")
    if results:
        scores = results["scores"]
        indices = results["indices"]
        q_len = results["q_len"]
        st.subheader("Query selector")
        q_idx = st.slider("Query index", 0, q_len-1, 0)

        topk = indices[q_idx]
        topscores = scores[q_idx]
        st.write({"query_index": int(q_idx)})
        st.table({"rank": list(range(1, len(topk)+1)), "db_idx": topk.tolist(), "score": topscores.tolist()})

        # Optional labels
        try:
            db_desc_len = results["db_len"]
            db_labels = safe_load_labels(db_labels_path)
            query_labels = safe_load_labels(query_labels_path)
            if db_labels is not None and query_labels is not None:
                if len(db_labels) != db_desc_len or len(query_labels) != q_len:
                    st.warning("Label lengths do not match descriptor counts; skipping recalls")
                else:
                    matches = [int(db_labels[i] == query_labels[q_idx]) for i in topk]
                    st.write("Label matches for this query:", matches)
        except Exception as e:
            st.warning(f"Label handling issue: {e}")

        # Visualize sequences (best-effort assumption: descriptor index maps to sequence start with given stride)
        if q_video_frames:
            st.subheader("Query sequence frames")
            q_seq = slice_sequence(q_video_frames, int(q_idx), int(seq_length), int(stride))
            if q_seq:
                cols = st.columns(min(len(q_seq), 6))
                for i, f in enumerate(q_seq):
                    cols[i % len(cols)].image(f, caption=f"Q seq frame {i}", width=300)
            else:
                st.info("No query frames available for this index")

        if db_video_frames:
            st.subheader("Top match sequence frames (rank 1)")
            top_db_idx = int(topk[0])
            db_seq = slice_sequence(db_video_frames, top_db_idx, int(seq_length), int(stride))
            if db_seq:
                cols = st.columns(min(len(db_seq), 6))
                for i, f in enumerate(db_seq):
                    cols[i % len(cols)].image(f, caption=f"DB rank1 frame {i}", width=300)
            else:
                st.info("No database frames available for this index")

        # Save results to a temp file for quick export
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp)/"topk_indices.npy", indices)
            np.save(Path(tmp)/"topk_scores.npy", scores)
            summary = {
                "metric": results["metric"],
                "k": results["k"],
                "db_vectors": int(results["db_len"]),
                "query_vectors": int(results["q_len"]),
                "dim": int(results["dim"]),
            }
            (Path(tmp)/"summary.json").write_text(json.dumps(summary, indent=2))
            st.download_button(
                label="Download summary.json",
                data=(Path(tmp)/"summary.json").read_bytes(),
                file_name="summary.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
