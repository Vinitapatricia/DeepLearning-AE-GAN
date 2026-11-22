import json, sys
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.config import set_seed, DEVICE, NOISE_DIM
from src.data import get_loader
from src.eval.eval import load
from src.models.vanillaAE import VanillaAE
from src.models.sparseAE import SparseAE
from src.models.vae import VAE
from src.models import gan, dcgan
from src.models.ddpm import UNet, Diffusion
import os

OUT = "outputs/figures"
LAT = 64


def show(ax, imgs, title, nrow):
    g = make_grid(imgs.cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=1)
    ax.imshow(g.permute(1, 2, 0), cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def get_models():
    return dict(
        ae=load(VanillaAE(), "outputs/checkpoints/vanilla_ae.pt"),
        sp=load(SparseAE(), "outputs/checkpoints/sparse_ae_1e-05.pt"),
        vae=load(VAE(), "outputs/checkpoints/vae.pt"),
        G1=load(gan.Generator(NOISE_DIM), "outputs/checkpoints/gan_G.pt"),
        G2=load(dcgan.Generator(NOISE_DIM), "outputs/checkpoints/dcgan_G.pt"),
    )

def fig1(m, X):
    with torch.no_grad():
        x = X[:8].to(DEVICE)
        rows = [x, m["ae"](x), m["sp"](x)[0], m["vae"].decode(m["vae"].encode(x)[0])]

    names = ["Ground truth", "Vanilla AE", "Sparse AE", "VAE"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 6))

    for ax, r, n in zip(axes, rows, names):
        show(ax, r, n, nrow=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_reconstruction.png", dpi=200)

def fig2(m):
    set_seed()
    with torch.no_grad():
        unet = load(UNet(), "outputs/checkpoints/ddpm.pt")
        diff = Diffusion(1000, DEVICE)
        panels = {
            "A: VAE": m["vae"].decode(torch.randn(16, LAT, device=DEVICE)),
            "B: Vanilla GAN": (m["G1"](torch.randn(16, NOISE_DIM, device=DEVICE)) + 1) / 2,
            "C: DCGAN": (m["G2"](torch.randn(16, NOISE_DIM, device=DEVICE)) + 1) / 2,
            "D: DDPM": (diff.sample(unet, 16, DEVICE) + 1) / 2,
        }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (n, im) in zip(axes, panels.items()):
        show(ax, im, n, nrow=4)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_samples.png", dpi=200)
    plt.close(fig)


def fig4(m, X, Y):
    ia = (Y == 1).nonzero()[0].item()
    ib = (Y == 9).nonzero()[0].item()
    xa, xb = X[ia:ia + 1].to(DEVICE), X[ib:ib + 1].to(DEVICE)
    alphas = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=DEVICE)[:, None]

    def interp(za, zb):
        return (1 - alphas) * za + alphas * zb

    with torch.no_grad():
        ae, sp, vae = m["ae"], m["sp"], m["vae"]
        rows = {
            "Vanilla AE": ae.decode(interp(ae.encode(xa), ae.encode(xb))),
            "Sparse AE": sp.decode(interp(sp.encode(xa)[0], sp.encode(xb)[0])),
            "VAE (mu)": vae.decode(interp(vae.encode(xa)[0], vae.encode(xb)[0])),
        }

        torch.manual_seed(0)
        z1, z2 = torch.randn(1, NOISE_DIM, device=DEVICE), torch.randn(1, NOISE_DIM, device=DEVICE)
        rows["Vanilla GAN"] = (m["G1"](interp(z1, z2)) + 1) / 2
        rows["DCGAN"] = (m["G2"](interp(z1, z2)) + 1) / 2

    fig, axes = plt.subplots(5, 1, figsize=(9, 8))
    for ax, (n, im) in zip(axes, rows.items()):
        show(ax, im, n, nrow=6)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_interpolation.png", dpi=200)
    plt.close(fig)


def fig5():
    files = {"Vanilla AE": "loss_vanillaAE", "Sparse AE": "loss_sparseAE_1e-05","VAE": "loss_vae", "Vanilla GAN": "loss_gan", "DCGAN": "loss_dcgan","DDPM": "loss_ddpm"}
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (n, f) in zip(axes.ravel(), files.items()):
        with open(f"outputs/{f}.json") as fp:
            d = json.load(fp)
        for k, v in d.items():
            if k != "epoch_time":
                ax.plot(range(1, len(v) + 1), v, label=k)
        ax.set_title(n)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5a_loss_curves.png", dpi=200)
    plt.close(fig)

    set_seed()
    with torch.no_grad():
        unet = load(UNet(), "outputs/checkpoints/ddpm.pt")
        diff = Diffusion(1000, DEVICE)
        _, traj = diff.sample(unet, 4, DEVICE, return_traj=True, traj_every=100)

    steps = len(traj)
    fig, axes = plt.subplots(4, steps, figsize=(steps * 1.3, 5.5))
    for j, snap in enumerate(traj):
        for i in range(4):
            axes[i, j].imshow((((snap[i, 0] + 1) / 2).clamp(0, 1)).cpu(), cmap="gray")
            axes[i, j].axis("off")
        axes[0, j].set_title(f"t={1000 - j * 100 if j else 1000}", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5b_ddpm_trajectory.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    set_seed()
    _, _, test = get_loader()
    X = torch.cat([x for x, _ in test])
    Y = torch.cat([y for _, y in test])
    which = sys.argv[1:] or ["1", "2", "4", "5"]
    m = get_models()
    if "1" in which:
        fig1(m, X)
    if "2" in which:
        fig2(m)
    if "4" in which:
        fig4(m, X, Y)
    if "5" in which:
        fig5()