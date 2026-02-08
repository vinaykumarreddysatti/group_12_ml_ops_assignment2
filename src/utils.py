import gzip
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = "./data/mnist.pkl.gz"

def load_mnist():
    with gzip.open(DATA_PATH, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    X_train, y_train = train_set
    X_test, y_test = test_set

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return (X_train, y_train), (X_test, y_test)


def load_data(num_clients):
    (X_train, y_train), _ = load_mnist()

    size = len(X_train) // num_clients
    client_loaders = []

    for i in range(num_clients):
        X_part = X_train[i*size:(i+1)*size]
        y_part = y_train[i*size:(i+size*(i+1))]
        dataset = TensorDataset(X_part, y_part)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        client_loaders.append(loader)

    return client_loaders


def evaluate(model):
    _, (X_test, y_test) = load_mnist()

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=1000
    )

    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

    return correct / len(X_test)
