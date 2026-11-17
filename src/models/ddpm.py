import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.linear1 = nn.Linear(n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(n_channels, n_channels)

    def forward(self, t):
        half = self.n_channels // 8
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        return self.linear2(self.act(self.linear1(emb)))

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, groups=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.drop = nn.Dropout(dropout)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.t_proj(nn.functional.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(nn.functional.silu(self.norm2(h))))
        return h + self.skip(x)

class Attention(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k * self.scale, dim=-1)
        h = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.out(h)


class UNet(nn.Module):
    def __init__(self, base=64, t_dim=256):
        super().__init__()
        c1, c2 = base, base * 2
        self.time_emb = TimeEmbedding(t_dim)
        self.inp = nn.Conv2d(1, c1, 3, padding=1)

        self.down1 = ResBlock(c1, c1, t_dim)
        self.down2 = ResBlock(c1, c2, t_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid1 = ResBlock(c2, c2, t_dim)
        self.mid_attn = Attention(c2)
        self.mid2 = ResBlock(c2, c2, t_dim)

        self.up2 = ResBlock(c2 + c2, c1, t_dim)
        self.up1 = ResBlock(c1 + c1, c1, t_dim)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x, t):
        te = self.time_emb(t)
        x0 = self.inp(x)
        d1 = self.down1(x0, te)
        d2 = self.down2(self.pool(d1), te)

        m = self.mid1(self.pool(d2), te)
        m = self.mid_attn(m)
        m = self.mid2(m, te)

        u2 = nn.functional.interpolate(m, scale_factor=2)
        u2 = self.up2(torch.cat([u2, d2], 1), te)
        u1 = nn.functional.interpolate(u2, scale_factor=2)
        u1 = self.up1(torch.cat([u1, d1], 1), te)
        return self.out(nn.functional.silu(self.out_norm(u1)))

class Diffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def sample(self, model, n, device, return_traj=False, traj_every=100):
        with torch.no_grad():

            x = torch.randn(n, 1, 28, 28, device=device)
            traj = [x.cpu()]
            for i in reversed(range(self.T)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                eps = model(x, t)
                alpha, abar, beta = self.alphas[i], self.abar[i], self.betas[i]
                mean = (x - beta / (1 - abar).sqrt() * eps) / alpha.sqrt()
                x = mean + beta.sqrt() * torch.randn_like(x) if i > 0 else mean
                if return_traj and i % traj_every == 0:
                    traj.append(x.cpu())
            x = x.clamp(-1, 1)
        return (x, traj) if return_traj else x