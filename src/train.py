import argparse, yaml, torch
from torch.utils.data import DataLoader
from models.tensorggraphrag import TensorGraphRAG
# ... dataset class omitted for brevity ...

def train(config):
    model = TensorGraphRAG(config).to(config["device"])
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # TODO: load DataLoader of (query, passages, answer)
    for epoch in range(config["epochs"]):
        model.train()
        for batch in dataloader:
            optim.zero_grad()
            # forward + loss (QA + latent-link BCE + sparsity)
            loss.backward()
            optim.step()
        # validation & checkpoint
    torch.save(model.state_dict(), "checkpoints/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)
