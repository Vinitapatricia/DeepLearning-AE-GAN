import os, json, time
import torch
import torch.nn as nn
from src.config import set_seed, DEVICE
from src.data import get_loader
from src.models.ddpm import UNet, Diffusion

def main(epochs=20, lr=1e-3, T= 1000):
    os.makedirs("outputs/checkpoints", exist_ok=True)
    set_seed()
    train_data, _, _ = get_loader()
    model = UNet().to(DEVICE)
    diff = Diffusion(T, DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    times = []

    for i in range(epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        n = 0

        for x, _ in train_data:
            x = x.to(DEVICE) * 2 - 1
            opt.zero_grad()

            t = torch.randint(0, T, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)
            loss = nn.functional.mse_loss(model(diff.q_sample(x, t, noise), t), noise)
            loss.backward()

            opt.step()

            run += loss.item() * x.size(0)
            n += x.size(0)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(run / n)

        print(f"epoch {i+1}: loss {losses[-1]:.4f} | {times[-1]:.1f}s")
        torch.save(model.state_dict(), "outputs/checkpoints/ddpm.pt")

    json.dump({"loss": losses, "epoch_time": times}, open("outputs/loss_ddpm.json", "w"))

if __name__ == "__main__":
    main()