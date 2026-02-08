import torch

def topk_compress(tensor, ratio=0.1):
    flat = tensor.view(-1)
    k = int(len(flat) * ratio)

    if k == 0:
        return torch.zeros_like(tensor)

    indices = torch.topk(flat.abs(), k).indices
    compressed = torch.zeros_like(flat)
    compressed[indices] = flat[indices]

    return compressed.view(tensor.shape)
