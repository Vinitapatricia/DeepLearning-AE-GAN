# DeepLearning-AE-GAN

Tugas Take-Home Deep Learning (D6421) Sesi 07: Generative Models.
Implementasi, pelatihan, dan evaluasi enam model generatif (Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM) pada Fashion-MNIST, beserta eksperimen pembuktian bahwa MLP Classifier tidak dapat dipakai untuk image generation.

- Nama: Vinita Patricia
- NIM: 2802417951
- Kelas: LB01

## Isi

| Model | File model | Skrip training | Loss |
|---|---|---|---|
| MLP Classifier (784-256-64-10) | `src/models/mlp.py` | `src/train_MLP.py` | Cross-Entropy |
| Vanilla AE | `src/models/vanillaAE.py` | `src/train_vanilla_ae.py` | MSE |
| Sparse AE | `src/models/sparseAE.py` | `src/train_sparse_ae.py` | MSE + λ·‖h‖₁ |
| VAE | `src/models/vae.py` | `src/train_vae.py` | BCE (sum) + KL |
| Vanilla GAN | `src/models/gan.py` | `src/train_gan.py` | BCE, non-saturating |
| DCGAN | `src/models/dcgan.py` | `src/train_dcgan.py` | BCE, non-saturating |
| DDPM | `src/models/ddpm.py` | `src/train_ddpm.py` | MSE pada noise ε |

Skrip lain:
- `src/config.py`: konstanta global (seed 42, batch size 128, device, dimensi noise).
- `src/data.py`: memuat Fashion-MNIST dan membagi data (stratified 80/10/10, seed 42; 56.000/7.000/7.000).
- `src/exp_mlp_limits.py`: eksperimen lima tahap pembuktian batasan MLP.
- `src/eval/metrics.py`: MSE, PSNR, SSIM.
- `src/eval/run_eval.py`: evaluasi gabungan (metrik rekonstruksi, FID, latency, waktu training per epoch).
- `src/make_figures.py`: membuat lima figur wajib.

## Persiapan

```bash
git clone https://github.com/Vinitapatricia/DeepLearning-AE-GAN.git
cd DeepLearning-AE-GAN
pip install -r requirements.txt
```

Semua perintah di bawah dijalankan dari **folder root repositori**.

## Cara menjalankan

Urutan yang disarankan:

```bash
# 1. Latih semua model
python -m src.train_MLP
python -m src.train_vanilla_ae
python -m src.train_sparse_ae 1e-5 # argumen = koefisien sparsity λ
python -m src.train_vae
python -m src.train_gan
python -m src.train_dcgan
python -m src.train_ddpm

# 2. Eksperimen pembuktian batasan MLP -> outputs/metrics_mlp_limits.json
python -m src.exp_mlp_limits

# 3. Evaluasi gabungan -> outputs/metrics_all.json
python -m src.eval.eval
```

Catatan:
- Evaluasi DDPM paling lama (sampling 1000 langkah, FID dengan 5000 sampel butuh sekitar 20 menit di GPU Colab).
- Semua training memakai Adam, 20 epoch, batch size 128, seed 42. Learning rate 1e-3 untuk semua model; GAN dan DCGAN memakai β = (0,5; 0,999).

## Hasil

| Model | Dim laten | Train s/epoch | Latency ms/batch 64 | MSE | PSNR (dB) | SSIM | FID (n=5000) |
|---|---|---|---|---|---|---|---|
| Vanilla AE | 64 | 1,32 | 0,246 | 0,00698 | 22,63 | 0,854 | N/A |
| Sparse AE (λ=1e-5) | 64 (h=256) | 1,43 | 0,204 | 0,00756 | 22,28 | 0,845 | N/A |
| VAE | 64 | 1,69 | 0,175 | 0,01345 | 19,52 | 0,761 | 96,25 |
| Vanilla GAN | 100 | 2,86 | 0,288 | N/A | N/A | N/A | 154,57 |
| DCGAN | 100 | 6,62 | 0,445 | N/A | N/A | N/A | 23,93 |
| DDPM | 28×28 | 36,06 | 13.720,6 | N/A | N/A | N/A | 20,35 |


Eksperimen MLP (z_MLP berdimensi 64):

| Eksperimen | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| Proyeksi transpos bobot dari z_MLP | 0,2857 | 5,48 | 0,054 |
| Frozen latent decoder dari z_MLP | 0,01769 | 18,25 | 0,706 |
| Vanilla AE (pembanding) | 0,00698 | 22,63 | 0,854 |

Akurasi test MLP Classifier: 89,4%.
