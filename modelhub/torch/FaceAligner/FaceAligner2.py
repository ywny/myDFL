from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xlib.file import SplittedFile
from xlib.torch import TorchDeviceInfo, get_cpu_device_info
from xlib.torch.model import XSegNet, MobileNet

class FaceAlignerNet(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        self.seg = XSegNet(in_ch, 16)
        self.mb = MobileNet(16, 4)

    def forward(self, inp):
        x = inp
        x = self.seg(x)

        shape = x.shape
        x = heatmaps_t = torch.sigmoid(x)
        #x = x.view(shape[0], shape[1], -1)
        #x = F.softmax(x / 0.1, dim=-1)
        #x = heatmaps_t = x.view(*shape)

        x = self.mb(x)

        scale_t, angle_t, tx_t, ty_t = torch.split(x, 1, -1)

        aff_t = torch.cat([torch.cos(angle_t)*scale_t, -torch.sin(angle_t)*scale_t, tx_t,
                           torch.sin(angle_t)*scale_t,  torch.cos(angle_t)*scale_t, ty_t,
                          ], dim=-1).view(-1,2,3)

        # from xlib.console import diacon
        # diacon.Diacon.stop()
        # import code
        # code.interact(local=dict(globals(), **locals()))

        return aff_t, heatmaps_t



# class CTSOT:
#     def __init__(self, device_info : TorchDeviceInfo = None,
#                        state_dict : Union[dict, None] = None,
#                        training : bool = False):
#         if device_info is None:
#             device_info = get_cpu_device_info()
#         self.device_info = device_info

#         self._net = net = CTSOTNet()

#         if state_dict is not None:
#             net.load_state_dict(state_dict)

#         if training:
#             net.train()
#         else:
#             net.eval()

#         self.set_device(device_info)

#     def set_device(self, device_info : TorchDeviceInfo = None):
#         if device_info is None or device_info.is_cpu():
#             self._net.cpu()
#         else:
#             self._net.cuda(device_info.get_index())

#     def get_state_dict(self):
#         return self.net.state_dict()

#     def get_net(self) -> CTSOTNet:
#         return self._net
