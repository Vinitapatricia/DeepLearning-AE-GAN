import json, time
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from src.config import set_seed, DEVICE, NOISE_DIM
from src.data import get_loader
from src.eval.metrics import bikin_metrics
from src.models.vanillaAE import VanillaAE
from src.models.sparseAE import SparseAE
from src.models.vae import VAE
from src.models import gan, dcgan
from src.models.ddpm import UNet, Diffusion

N_FID = 5000
BATCH = 64

def load(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()

def to_uint8_rgb(x01):
    return (x01.clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)

def fid_score(real_loader, gen_fn, n):
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    with torch.no_grad():
        for x, _ in real_loader:
            fid.update(to_uint8_rgb(x).to(DEVICE), real=True)
        done = 0
        while done < n:
            b = min(BATCH, n - done)
            fid.update(to_uint8_rgb(gen_fn(b)).to(DEVICE), real=False)
            done += b
        score = fid.compute().item()
    return score

def latency(gen_fn, repeats=10, warmup=2):
    with torch.no_grad():
        for _ in range(warmup):
            gen_fn(BATCH)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(repeats):
            gen_fn(BATCH)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    return (t1 - t0) / repeats * 1000

def main():
    set_seed()
    _, _, test = get_loader()
    X = torch.cat([x for x, _ in test])
    Xd = X.to(DEVICE)
    res = {}

    ae = load(VanillaAE(), "outputs/checkpoints/vanilla_ae.pt")
    sp = load(SparseAE(), "outputs/checkpoints/sparse_ae_1e-05.pt")
    vae = load(VAE(), "outputs/checkpoints/vae.pt")
    with torch.no_grad():
        res["Vanilla AE"] = bikin_metrics(X, ae(Xd).cpu())
        res["Sparse AE"] = bikin_metrics(X, sp(Xd)[0].cpu())
        res["VAE"] = bikin_metrics(X, vae.decode(vae.encode(Xd)[0]).cpu())

    vanilla_G = load(gan.Generator(NOISE_DIM), "outputs/checkpoints/gan_G.pt")
    dc_G = load(dcgan.Generator(NOISE_DIM), "outputs/checkpoints/dcgan_G.pt")
    unet = load(UNet(), "outputs/checkpoints/ddpm.pt")
    diff = Diffusion(1000, DEVICE)
    laten = 64

    gens = {
        "VAE": lambda b: vae.decode(torch.randn(b, laten, device=DEVICE)),
        "Vanilla GAN": lambda b: (vanilla_G(torch.randn(b, NOISE_DIM, device=DEVICE)) + 1) / 2,
        "DCGAN": lambda b: (dc_G(torch.randn(b, NOISE_DIM, device=DEVICE)) + 1) / 2,
        "DDPM": lambda b: (diff.sample(unet, b, DEVICE) + 1) / 2,
    }
    n_fid = {"VAE": N_FID, "Vanilla GAN": N_FID, "DCGAN": N_FID, "DDPM": N_FID}

    for name, fn in gens.items():
        print("FID", name)
        res.setdefault(name, {})
        res[name]["fid"] = fid_score(test, fn, n_fid[name])
        res[name]["latency_ms_b64"] = latency(fn, repeats=3 if name == "DDPM" else 10)
        res[name]["n_fid"] = n_fid[name]
        print(name, res[name])

    for name, m, f in [("Vanilla AE", ae, lambda b: ae(Xd[:b])), ("Sparse AE", sp, lambda b: sp(Xd[:b])[0])]:
        res[name]["latency_ms_b64"] = latency(f)

    for name, path in [("Vanilla AE", "loss_vanillaAE"),
                       ("Sparse AE", "loss_sparseAE_1e-05"),
                       ("VAE", "loss_vae"),
                       ("Vanilla GAN", "loss_gan"),
                       ("DCGAN", "loss_dcgan"),
                       ("DDPM", "loss_ddpm")]:
        try:
            t = json.load(open(f"outputs/{path}.json"))["epoch_time"]
            res.setdefault(name, {})["train_s_per_epoch"] = sum(t) / len(t)
        except FileNotFoundError:
            print("JSON tidak ada:", path)

    json.dump(res, open("outputs/metrics_all.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
