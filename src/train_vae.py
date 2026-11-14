import os, json, time
import torch
import torch.nn as nn
from src.config import set_seed, DEVICE
from src.data import get_loader
from src.models.vae import VAE

def main(epochs=20, lr=1e-3):
    os.makedirs("outputs/checkpoints", exist_ok=True)
    set_seed()
    train_data, val_data, test_data = get_loader()
    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = 0.0
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

            x_hat, mu, logvar = model(x)
            bce = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = bce + kl
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
                correct += nn.functional.mse_loss(model(x), x).item() * x.size(0)
                total += x.size(0)
            val_acc = correct / total

        print(f"epoch {i+1}: loss {loss.item()} | val acc {val_acc} | {times[-1]:.2f}s")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "outputs/checkpoints/vae.pt")

    json.dump({"loss": train_losses, "epoch_time": times}, open("outputs/loss_vae.json", "w"))

if __name__ == "__main__":
    main()