import os, json, time
import torch
import torch.nn as nn
from src.config import set_seed, DEVICE, NOISE_DIM
from src.data import get_loader
from src.models.dcgan import Generator, Discriminator, init_weights

def main(epochs=20, lr=1e-3):
    os.makedirs("outputs/checkpoints", exist_ok=True)
    set_seed()
    train_data, _, _ = get_loader()
    g_model = Generator(NOISE_DIM).to(DEVICE)
    g_model.apply(init_weights)
    d_model = Discriminator().to(DEVICE)
    d_model.apply(init_weights)
    optG = torch.optim.Adam(g_model.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(d_model.parameters(), lr=lr, betas=(0.5, 0.999))
    lossG_hist = []
    lossD_hist = []
    times = []

    for i in range(epochs):
        t0 = time.time()
        running_g = 0.0
        running_d = 0.0
        n = 0.0

        for x, _ in train_data:
            x = (x.to(DEVICE) * 2 - 1)
            b = x.size(0)
            real = torch.ones(b, 1, device=DEVICE)
            fake = torch.zeros(b, 1, device=DEVICE)

            # update D
            optD.zero_grad()
            z = torch.randn(b, NOISE_DIM, device=DEVICE)
            x_fake = g_model(z)

            lossD = (nn.functional.binary_cross_entropy_with_logits(d_model(x), real)) + (nn.functional.binary_cross_entropy_with_logits(d_model(x_fake.detach()), fake))
            lossD.backward()

            optD.step()

            # update G
            optG.zero_grad()

            lossG = nn.functional.binary_cross_entropy_with_logits(d_model(x_fake), real)
            lossG.backward()

            optG.step()

            running_g += lossG.item() * b
            running_d += lossD.item() * b
            n += b

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        lossG_hist.append(running_g / n)
        lossD_hist.append(running_d / n)

        print(f"epoch {i+1}: D {lossD_hist[-1]} | G {lossG_hist[-1]} | {times[-1]:.2f}s")

    torch.save(g_model.state_dict(), "outputs/checkpoints/dcgan_G.pt")
    torch.save(d_model.state_dict(), "outputs/checkpoints/dcgan_D.pt")
    json.dump({"loss_G": lossG_hist, "loss_D": lossD_hist, "epoch_time": times}, open("outputs/loss_dcgan.json", "w"))


if __name__ == "__main__":
    main()
                 