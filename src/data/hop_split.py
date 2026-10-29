"""Construct 1–4 hop subsets from HotpotQA supporting-fact annotations."""
import argparse, json
from collections import defaultdict
from pathlib import Path

def hop_length(supporting_facts):
    """Shortest supporting-fact chain length (number of distinct titles)."""
    return len(set(t for t, _ in supporting_facts))

def main(args):
    with open(args.input) as f:
        data = json.load(f)
    hops = defaultdict(list)
    for ex in data:
        h = hop_length(ex.get("supporting_facts", []))
        h = max(1, min(4, h))          # clamp to 1–4
        hops[h].append(ex)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for h, items in hops.items():
        with open(out / f"hop_{h}.json", "w") as f:
            json.dump(items, f)
        print(f"Hop {h}: {len(items)} examples")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args)
