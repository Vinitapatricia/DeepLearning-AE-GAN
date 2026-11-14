import torch
from torchmetrics.functional import structural_similarity_index_measure

def bikin_metrics(x, x_hat):
    x_hat = x_hat.clamp(0, 1)
    mse_img = ((x - x_hat) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    psnr = (10 * torch.log10(1.0 / mse_img)).mean().item()
    ssim = structural_similarity_index_measure(x_hat, x, data_range=1.0).item()
    return {"mse": mse_img.mean().item(), "psnr": psnr, "ssim": ssim}