import torch
from src.compression import topk_compress

def local_train(model, data_loader, epochs=1, lr=0.01):
    model = model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for _ in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    return model


def get_model_update(global_model, local_model, compression_ratio):
    update = {}
    total_params = 0
    sent_params = 0

    for key in global_model.state_dict():
        diff = local_model.state_dict()[key] - global_model.state_dict()[key]
        compressed = topk_compress(diff, compression_ratio)

        update[key] = compressed

        total_params += diff.numel()
        sent_params += (compressed != 0).sum().item()

    return update, sent_params, total_params
