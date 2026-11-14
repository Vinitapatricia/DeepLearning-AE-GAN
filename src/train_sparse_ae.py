import os, json, time
import sys
import torch
import torch.nn as nn
from src.config import set_seed, DEVICE
from src.data import get_loader
from src.models.sparseAE import SparseAE

LAMBDA_SPARSE = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
tag = f"{LAMBDA_SPARSE:g}"

def main(epochs=20, lr=1e-3):
    os.makedirs("outputs/checkpoints", exist_ok=True)
    set_seed()
    train_data, val_data, test_data = get_loader()
    model = SparseAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    train_losses = []
    times = []

    for i in range(epochs):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()

        for x, _ in train_data:
            x = x.to(DEVICE)
            
            optimizer.zero_grad()

            x_hat, h = model(x)
            loss = nn.functional.mse_loss(x_hat, x) + LAMBDA_SPARSE * h.abs().mean(0).sum()
            loss.backward()
            
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        train_losses.append(running / n)

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, _ in val_data:
                x= x.to(DEVICE)
                x_hat, h = model(x)
                loss_val = nn.functional.mse_loss(x_hat, x) + LAMBDA_SPARSE * h.abs().mean(0).sum()
                correct += loss_val.item() * x.size(0)
                total += x.size(0)
            val_loss = correct / total

        print(f"epoch {i+1}: loss {train_losses[-1]} | val loss {val_loss} | {times[-1]:.2f}s")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"outputs/checkpoints/sparse_ae_{tag}.pt")

    json.dump({"loss": train_losses, "epoch_time": times}, open(f"outputs/loss_sparseAE_{tag}.json", "w"))

if __name__ == "__main__":
    main()