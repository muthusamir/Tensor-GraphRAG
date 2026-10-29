"""Passage segmentation and embedding for HotpotQA / NQ / MultiRC."""
import argparse, json, os
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

def load_hotpotqa(path):
    with open(path) as f:
        return json.load(f)

def segment_passages(docs, max_len=120):
    passages = []
    for doc in docs:
        text = doc if isinstance(doc, str) else " ".join(doc)
        words = text.split()
        for i in range(0, len(words), max_len):
            passages.append(" ".join(words[i:i+max_len]))
    return passages

def main(args):
    model = SentenceTransformer(args.encoder)
    raw = load_hotpotqa(args.input)
    passages, meta = [], []
    for ex in tqdm(raw):
        ctx = ex.get("context", [])
        segs = segment_passages([c[1] for c in ctx] if ctx else [ex.get("passage","")])
        for s in segs:
            passages.append(s)
            meta.append({"id": ex["_id"], "title": ex.get("title","")})
    emb = model.encode(passages, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    with open(out / "passages.json", "w") as f:
        json.dump({"passages": passages, "meta": meta}, f)
    print(f"Saved {len(passages)} passages → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    args = p.parse_args()
    main(args)
