import torch
from thop import profile


def FLOPs_and_Params(model, size, device):
    min_size = size
    max_size = int(round(1333 / 800 * size))
    x = torch.randn(1, 3, min_size, max_size).to(device)

    flops, params = profile(model, inputs=(x, ))
    print('FLOPs : ', flops / 1e9, ' B')
    print('Params : ', params / 1e6, ' M')


if __name__ == "__main__":
    pass
