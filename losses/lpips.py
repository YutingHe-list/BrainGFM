"""感知损失。结构与原来的 LPIPS 一致，权重读本地文件，不联网下载。"""
import os
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from losses.util import REPO_ROOT

VGG16_LOCAL_PATH = os.path.join(REPO_ROOT, "pretrained", "vgg16-397923af.pth")
VGG_LPIPS_PATH = os.path.join(REPO_ROOT, "pretrained", "vgg.pth")


class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_state_dict(torch.load(VGG_LPIPS_PATH, map_location="cpu"), strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        in0, in1 = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0), self.net(in1)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        val = 0.0
        for kk in range(len(self.chns)):
            f0 = normalize_tensor(outs0[kk])
            f1 = normalize_tensor(outs1[kk])
            diff = (f0 - f1) ** 2
            val = val + spatial_average(lins[kk].model(diff), keepdim=True)
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        base = models.vgg16(pretrained=False)
        sd = torch.load(VGG16_LOCAL_PATH, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        base.load_state_dict(sd)
        feats = base.features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), feats[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), feats[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), feats[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), feats[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), feats[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def normalize_tensor(x, eps=1e-10):
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
