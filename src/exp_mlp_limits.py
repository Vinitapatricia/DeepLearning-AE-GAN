import json
import torch
import torch.nn.functional
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from src.config import set_seed, DEVICE, BATCH_SIZE
from src.data import get_loader
from src.models.mlp import MLP, MLPDecoder
from src.eval.metrics import bikin_metrics

def extract(classifier, loader):
    with torch.no_grad():
        classifier.eval()
        xs = []
        zs = []
        for x, y in loader:
            xs.append(x)
            zs.append(classifier.features(x.to(DEVICE)).cpu())
        return torch.cat(xs), torch.cat(zs)

def row(ax, imgs, title):
    g = make_grid(imgs.clamp(0, 1), nrow=len(imgs), padding=2, pad_value=1)
    ax.imshow(g.permute(1, 2, 0)); ax.set_title(title, fontsize=9); ax.axis("off")

def main():
    set_seed()
    train_dataset, val_dataset, test_dataset = get_loader()

    classifier = MLP().to(DEVICE)
    classifier.load_state_dict(torch.load("outputs/checkpoints/mlp_classifier.pt"))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    W1 = classifier.fc1.weight.cpu()
    W2 = classifier.fc2.weight.cpu()
    W3 = classifier.fc3.weight.cpu()
    X_train, Z_train = extract(classifier, train_dataset)
    X_test, Z_test =extract(classifier, test_dataset)
    results = {}

    print("rank(W1^T W1) =", torch.linalg.matrix_rank(W1.T @ W1).item())
    print("||W2 W2^T - I||_F =", torch.norm(W2 @ W2.T - torch.eye(64)).item())


    y_c = torch.eye(10)
    z_hat = torch.nn.functional.relu(y_c @ W3)
    h1_hat = torch.nn.functional.relu(z_hat @ W2)
    x_cls = torch.sigmoid(h1_hat @ W1).view(10, 1, 28, 28)

    x_lat_all = torch.sigmoid(torch.nn.functional.relu(Z_test @ W2) @ W1).view(-1, 1, 28, 28)
    results["tahap3_latent_transpose"] = bikin_metrics(X_test, x_lat_all)

    decoder = MLPDecoder(64).to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    data_loader = DataLoader(TensorDataset(Z_train, X_train), batch_size=BATCH_SIZE, shuffle=True)

    for i in range(20):
        for z, x in data_loader:
            z, x = z.to(DEVICE), x.to(DEVICE)
            loss = torch.nn.functional.mse_loss(decoder(z), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"decoder epoch {i} | mse {loss.item()}")

    decoder.eval()
    with torch.no_grad():
        rec_te = decoder(Z_test.to(DEVICE)).cpu()

    results["tahap4_frozen_decoder_zMLP"] = bikin_metrics(X_test, rec_te)
    torch.save(decoder.state_dict(), "outputs/checkpoints/mlp_frozen_decoder.pt")


    mu, sigma = Z_test.mean(0), Z_test.std(0)
    z_rand = mu + sigma * torch.randn(16, 64)
    with torch.no_grad():
        x_rand = decoder(z_rand.to(DEVICE)).cpu()

    print(f"proporsi elemen z_rand negatif: {(z_rand < 0).float().mean():.2%}")


    json.dump(results, open("outputs/metrics_mlp_limits.json", "w"), indent=2)
    print(json.dumps(results, indent=2))

    fig, axes = plt.subplots(5, 1, figsize=(10, 9))
    row(axes[0], X_test[:8], "Ground truth")
    row(axes[1], x_cls, "W^T dari one-hot kelas y_c (kelas 0-9)")
    row(axes[2], x_lat_all[:8], "W^T dari z_MLP citra uji")
    row(axes[3], rec_te[:8], "Frozen decoder dari z_MLP ]")
    row(axes[4], x_rand[:8], "Prior sampling z ~ N(mu_z, diag(sigma_z^2))")
    plt.tight_layout()
    plt.savefig("outputs/figures/fig3_mlp_limits.png", dpi=200)
    



if __name__ == "__main__":
    main()