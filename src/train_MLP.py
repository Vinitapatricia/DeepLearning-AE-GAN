import torch
import torch.nn as nn
from src.config import set_seed, DEVICE
from src.data import get_loader
from src.models.mlp import MLPClassifier

def main(epochs=20, lr=1e-3):
    set_seed()
    train_data, val_data, test_data = get_loader()
    model = MLPClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cross_en = nn.CrossEntropyLoss()
    best_val = 0.0
    train_losses = []

    for i in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_data:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()

            loss = cross_en(model(x), y)
            loss.backward()
            
            optimizer.step()

            running += loss.item() * y.size(0)
            n += y.size(0)
        train_losses.append(running / n)

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_data:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
            val_acc = correct / total

        print(f"epoch {i+1}: loss {loss.item()} | val acc {val_acc}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "outputs/checkpoints/mlp_classifier.pt")


    model.load_state_dict(torch.load("outputs/checkpoints/mlp_classifier.pt"))

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
        test_acc = correct / total

    print(f"test acc: {test_acc}")

if __name__ == "__main__":
    main()
