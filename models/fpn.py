import torch.nn as nn
import torch.nn.functional as F


# Basic FPN
class BasicFPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channel=256):
        super(BasicFPN, self).__init__()
        c3, c4, c5 = in_channels
        # latter layers
        self.latter_1 = nn.Conv2d(c3, out_channel, kernel_size=1)
        self.latter_2 = nn.Conv2d(c4, out_channel, kernel_size=1)
        self.latter_3 = nn.Conv2d(c5, out_channel, kernel_size=1)

        # smooth layers
        self.smooth_1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.smooth_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.smooth_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, features):
        c3, c4, c5 = features
        # fpn
        p5 = self.latter_3(c5)
        p5_up = F.interpolate(p5, size=c4.shape[-2:])
        p5 = self.smooth_3(p5)

        p4 = self.latter_2(c4) + p5_up
        p4_up = F.interpolate(p4, size=c3.shape[-2:])
        p4 = self.smooth_2(p4)

        p3 = self.smooth_1(self.latter_1(c3) + p4_up)

        return [p3, p4, p5]


def build_fpn(model_name='basic_fpn', in_channels=[512, 1024, 2048], out_channel=256):
    if model_name == 'basic_fpn':
        print("Basic FPN ...")
        return BasicFPN(in_channels, out_channel)
    elif model_name == 'bifpn':
        print('BiFPN ...')
        return None
    else:
        print("Unknown FPN version ...")
        exit()
