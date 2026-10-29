import argparse, yaml, torch, json
from models.tensorggraphrag import TensorGraphRAG

def evaluate(config, checkpoint, split="dev"):
    model = TensorGraphRAG(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    # load data, run inference, compute EM / F1 / evidence metrics
    # hop-wise evaluation when --hop_eval is set
    # ablation when --ablation is set
    print("Evaluation finished.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hop_eval", action="store_true")
    p.add_argument("--ablation", action="store_true")
    args = p.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluate(config, args.checkpoint)
