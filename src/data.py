import torch, torchvision
import numpy as np
from torchvision.datasets import FashionMNIST
from src.config import SEED, BATCH_SIZE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def loader(x, y, shuffle):
  dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
  return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def get_loader():
    train_dataset = FashionMNIST("data", train = True, download=True)
    test_dataset = FashionMNIST("data", train = False, download=True)

    X = np.concatenate([train_dataset.data.numpy(), test_dataset.data.numpy()])
    Y = np.concatenate([train_dataset.targets.numpy(), test_dataset.targets.numpy()])

    X = (X.astype(np.float32) / 255.0)[:, None, :, :]
    x_train, x_tmp, y_train, y_tmp = train_test_split( X, Y, test_size=0.2, random_state=SEED, stratify=Y )
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    train_loader = loader(x_train, y_train, True)
    val_loader = loader(x_val, y_val, False)
    test_loader = loader(x_test, y_test, False)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train, val, test = get_loader()
    print(len(train.dataset), len(val.dataset), len(test.dataset))
    x, y = next(iter(train))
    print(x.shape, x.min().item(), x.max().item())
    print(np.bincount(test.dataset.tensors[1].numpy())) 