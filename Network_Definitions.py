import torch.nn as nn
import torch.nn.functional as F
import torch as torch
from torch.utils import data
import copy
import hexagdly as hg
import pandas as pd
import numpy as np

import GAN_Utils as GU

from matplotlib import pyplot as plt
import torch.optim as optim
import time as time

from sklearn import metrics as metrics


class BasicBlockQ3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(BasicBlockQ3, self).__init__()

        self.conv1 = hg.Conv2d(inplanes, planes, 1)  # conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU()

        self.conv2 = hg.Conv2d(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu_end = nn.PReLU()

        self.downsample = (inplanes != planes)
        self.stride = stride

        if self.downsample:
            self.conv3 = hg.Conv2d(planes, planes, 3, stride=2)
            self.bn3 = nn.BatchNorm2d(planes)
            self.relu2 = nn.PReLU()
            self.downsample_res = nn.Sequential(
                hg.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        # print("VC: ", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            # print("Sampling")
            # print("A: ",residual.shape)
            out = self.relu2(out)
            out = self.conv3(out)
            out = self.bn3(out)
            residual = self.downsample_res(residual)
            residual = nn.functional.interpolate(residual, size=out.shape[2:])
        # print("A: ", residual.shape)
        # print("B: ", out.shape)
        out += residual
        out = self.relu_end(out)

        return out


class HexNetDeep_q(nn.Module):
    def __init__(self):
        super(HexNetDeep_q, self).__init__()
        # size before x.view / batch size
        input_channel = 2
        self.size2 = 1200

        self.block11 = BasicBlockQ3(input_channel * 1, input_channel * 6, stride=1)
        self.block12 = BasicBlockQ3(input_channel * 6, input_channel * 6, stride=1)
        self.block13 = BasicBlockQ3(input_channel * 6, input_channel * 6, stride=1)
        self.block21 = BasicBlockQ3(input_channel * 6, input_channel * 12, stride=1)
        self.block22 = BasicBlockQ3(input_channel * 12, input_channel * 12, stride=1)
        self.block23 = BasicBlockQ3(input_channel * 12, input_channel * 12, stride=1)
        self.block31 = BasicBlockQ3(input_channel * 12, input_channel * 24, stride=1)
        self.block32 = BasicBlockQ3(input_channel * 24, input_channel * 24, stride=1)
        self.block33 = BasicBlockQ3(input_channel * 24, input_channel * 24, stride=1)

        self.fci = nn.Linear(self.size2, self.size2)
        self.fct = nn.Linear(self.size2, self.size2)

        self.block11t = BasicBlockQ3(input_channel * 1, input_channel * 6, stride=1)
        self.block12t = BasicBlockQ3(input_channel * 6, input_channel * 6, stride=1)
        self.block13t = BasicBlockQ3(input_channel * 6, input_channel * 6, stride=1)
        self.block21t = BasicBlockQ3(input_channel * 6, input_channel * 12, stride=1)
        self.block22t = BasicBlockQ3(input_channel * 12, input_channel * 12, stride=1)
        self.block23t = BasicBlockQ3(input_channel * 12, input_channel * 12, stride=1)
        self.block31t = BasicBlockQ3(input_channel * 12, input_channel * 24, stride=1)
        self.block32t = BasicBlockQ3(input_channel * 24, input_channel * 24, stride=1)
        self.block33t = BasicBlockQ3(input_channel * 24, input_channel * 24, stride=1)

        self.relu_i = nn.PReLU()
        self.relu_t = nn.PReLU()

        self.fc4 = nn.Linear(2 * self.size2, 2)

    def forward(self, x, y, vmax, vmin):
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        x = self.block31(x)
        x = self.block32(x)
        x = self.block33(x)
        # print("A: ", x.shape)
        x = x.view(-1, self.size2)
        # x = x.view(-1, self.size2)
        # print("A: ", x.shape)
        x = self.relu_i(self.fci(x))

        y = self.block11t(y)
        y = self.block12t(y)
        y = self.block13t(y)
        y = self.block21t(y)
        y = self.block22t(y)
        y = self.block23t(y)
        y = self.block31t(y)
        y = self.block32t(y)
        y = self.block33t(y)
        y = y.view(-1, self.size2)
        # y = y.view(-1, self.size2)
        # print("B: ", y.shape)
        y = self.relu_t(self.fct(y))

        x = torch.cat([x, y], dim=1)

        x = x.view(-1, 2 * self.size2)
        # print(x.shape)
        # print(vmax.shape)
        # x = torch.cat([x, vmax], dim=1)
        x = self.fc4(x)

        return x


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nonlinear_function=None):
        super(BasicBlock2, self).__init__()

        if nonlinear_function is None:
            nonlinear_function = nn.ReLU()

        self.conv1 = hg.Conv2d(inplanes, planes, 1)  # conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nonlinear_function
        self.conv2 = hg.Conv2d(planes, planes, 3, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.downsample = downsample
        self.stride = stride

        self.downsample = nn.Sequential(
            hg.Conv2d(inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # print("Sampling")
            # print("A: ",residual.shape)
            residual = self.downsample(residual)
            residual = nn.functional.interpolate(residual, size=out.shape[2:])
        # print("A: ", residual.shape)
        # print("B: ", out.shape)
        out += residual
        out = self.relu(out)

        return out


class HexNetDeep_res1_en1(nn.Module):
    def __init__(self):
        super(HexNetDeep_res1_en1, self).__init__()
        # size before x.view / batch size
        input_channel = 2

        self.nlf1 = nn.PReLU()
        self.nlf2 = nn.PReLU()
        self.nlf3 = nn.PReLU()
        self.nlf4 = nn.PReLU()
        self.nlf5 = nn.PReLU()

        self.nlf1t = nn.PReLU()
        self.nlf2t = nn.PReLU()
        self.nlf3t = nn.PReLU()
        self.nlf4t = nn.PReLU()
        self.nlf5t = nn.PReLU()

        self.block1 = BasicBlock2(input_channel * 1, input_channel * 4, 1, nonlinear_function=self.nlf1)
        self.block2 = BasicBlock2(input_channel * 4, input_channel * 8, 2, nonlinear_function=self.nlf2)
        self.block3 = BasicBlock2(input_channel * 8, input_channel * 16, 2, nonlinear_function=self.nlf3)
        self.block4 = BasicBlock2(input_channel * 16, input_channel * 32, 2, nonlinear_function=self.nlf4)
        self.block5 = BasicBlock2(input_channel * 32, input_channel * 64, 1, nonlinear_function=self.nlf5)

        self.block1t = BasicBlock2(input_channel * 1, input_channel * 4, 1, nonlinear_function=self.nlf1t)
        self.block2t = BasicBlock2(input_channel * 4, input_channel * 8, 2, nonlinear_function=self.nlf2t)
        self.block3t = BasicBlock2(input_channel * 8, input_channel * 16, 2, nonlinear_function=self.nlf3t)
        self.block4t = BasicBlock2(input_channel * 16, input_channel * 32, 2, nonlinear_function=self.nlf4t)
        self.block5t = BasicBlock2(input_channel * 32, input_channel * 64, 1, nonlinear_function=self.nlf5t)

        self.latent_w = 2078
        latent_w = self.latent_w
        self.ll_fc1 = nn.Linear(latent_w, latent_w)
        self.ll_fc2 = nn.Linear(latent_w, latent_w)
        self.ll_fc3 = nn.Linear(latent_w, latent_w)
        self.ll_fc4 = nn.Linear(latent_w, latent_w)
        self.ll_fc5 = nn.Linear(latent_w, latent_w)
        self.ll_fc6 = nn.Linear(latent_w, latent_w)

        self.nlf1ll = nn.PReLU()
        self.nlf2ll = nn.PReLU()
        self.nlf3ll = nn.PReLU()
        self.nlf4ll = nn.PReLU()
        self.nlf5ll = nn.PReLU()
        self.nlf6ll = nn.PReLU()

        self.size2 = 3200

        # self.fc1 = nn.Linear(2*self.size2, self.size2)
        # self.fc2 = nn.Linear(self.size2, self.size2)
        self.fc1 = nn.Linear(self.size2, self.size2)
        self.fc2 = nn.Linear(self.size2, self.size2)

        self.fc3 = nn.Linear(2 * self.size2 + self.latent_w, 2 * self.size2)
        self.fc4 = nn.Linear(2 * self.size2, 2)

        self.nlf1_fc = nn.PReLU()
        self.nlf2_fc = nn.PReLU()

        self.nlf3_fc = nn.PReLU()
        self.nlf4_fc = nn.PReLU()

        self.nlf = nn.LeakyReLU()
        # self.softmax = nn.Softmaxinfo_col_num(dim = 1) # Softmax is already included in CrossEntropyLoss
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, vmax, vmin):
        # vmax = 0
        # vmin = 0 ,vmax,vmin
        # print("A: ",q.shape)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(-1, self.size2)
        x = self.nlf1_fc(self.fc1(x))

        y = self.block1t(y)
        y = self.block2t(y)
        y = self.block3t(y)
        y = self.block4t(y)
        y = self.block5t(y)
        y = y.view(-1, self.size2)
        y = self.nlf2_fc(self.fc2(y))

        ls = torch.cat([x.view(-1, self.latent_w), y(-1, self.latent_w)])
        ls = self.nlf1ll(self.ll_fc1(ls))
        ls = self.nlf2ll(self.ll_fc2(ls))
        ls = self.nlf3ll(self.ll_fc3(ls))
        ls = self.nlf4ll(self.ll_fc4(ls))
        ls = self.nlf5ll(self.ll_fc5(ls))
        ls = self.nlf6ll(self.ll_fc6(ls))

        # x = self.conv3_5(x)
        # print("Q: ",x.shape)
        # print("Q: ",vmax.unsqueeze(1).shape)
        x = torch.cat([x, y, ls], dim=1)
        # print("Q: ",x.shape)
        x = x.view(-1, self.size2 * 2 + self.latent_w)

        # x = self.nlf(self.fc1(x))
        # x = self.nlf(self.fc2(x))
        x = self.nlf3_fc(self.fc3(x))
        x = self.nlf4_fc(self.fc4(x))

        return x


class HexNetDeep_res1_it(nn.Module):
    def __init__(self):
        super(HexNetDeep_res1_it, self).__init__()
        # size before x.view / batch size
        input_channel = 2

        self.nlf1 = nn.PReLU()
        self.nlf2 = nn.PReLU()
        self.nlf3 = nn.PReLU()
        self.nlf4 = nn.PReLU()
        self.nlf5 = nn.PReLU()

        self.nlf1t = nn.PReLU()
        self.nlf2t = nn.PReLU()
        self.nlf3t = nn.PReLU()
        self.nlf4t = nn.PReLU()
        self.nlf5t = nn.PReLU()

        self.block1 = BasicBlock2(input_channel * 1, input_channel * 4, 1, nonlinear_function=self.nlf1)
        self.block2 = BasicBlock2(input_channel * 4, input_channel * 8, 2, nonlinear_function=self.nlf2)
        self.block3 = BasicBlock2(input_channel * 8, input_channel * 16, 2, nonlinear_function=self.nlf3)
        self.block4 = BasicBlock2(input_channel * 16, input_channel * 32, 2, nonlinear_function=self.nlf4)
        self.block5 = BasicBlock2(input_channel * 32, input_channel * 64, 1, nonlinear_function=self.nlf5)

        self.block1t = BasicBlock2(input_channel * 1, input_channel * 4, 1, nonlinear_function=self.nlf1t)
        self.block2t = BasicBlock2(input_channel * 4, input_channel * 8, 2, nonlinear_function=self.nlf2t)
        self.block3t = BasicBlock2(input_channel * 8, input_channel * 16, 2, nonlinear_function=self.nlf3t)
        self.block4t = BasicBlock2(input_channel * 16, input_channel * 32, 2, nonlinear_function=self.nlf4t)
        self.block5t = BasicBlock2(input_channel * 32, input_channel * 64, 1, nonlinear_function=self.nlf5t)

        self.size2 = 3200

        # self.fc1 = nn.Linear(2*self.size2, self.size2)
        # self.fc2 = nn.Linear(self.size2, self.size2)
        self.fc1 = nn.Linear(self.size2, self.size2)
        self.fc2 = nn.Linear(self.size2, self.size2)

        self.fc3 = nn.Linear(2 * self.size2, 2 * self.size2)
        self.fc4 = nn.Linear(2 * self.size2, 2)

        self.nlf1_fc = nn.PReLU()
        self.nlf2_fc = nn.PReLU()

        self.nlf3_fc = nn.PReLU()
        self.nlf4_fc = nn.PReLU()

        self.nlf = nn.LeakyReLU()
        # self.softmax = nn.Softmaxinfo_col_num(dim = 1) # Softmax is already included in CrossEntropyLoss
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, vmax, vmin):
        inbs = x.shape[0]
        # vmax = 0
        # vmin = 0 ,vmax,vmin
        # print("A: ",q.shape)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # print("SU: ",x.shape)
        x = x.view(-1, self.size2)
        x = self.nlf1_fc(self.fc1(x))

        y = self.block1t(y)
        y = self.block2t(y)
        y = self.block3t(y)
        y = self.block4t(y)
        y = self.block5t(y)
        y = y.view(-1, self.size2)
        y = self.nlf2_fc(self.fc2(y))

        # x = self.conv3_5(x)
        # print("Q: ",x.shape)
        # print("Q: ",vmax.unsqueeze(1).shape)
        # print("FU: ",x.shape)
        # print("FU2: ",y.shape)
        x = torch.cat([x, y], dim=1)
        # print("Q: ",x.shape)
        x = x.view(-1, self.size2 * 2)
        # print("Q7: ",x.shape)
        # x = self.nlf(self.fc1(x))
        # x = self.nlf(self.fc2(x))
        x = self.nlf3_fc(self.fc3(x))
        x = self.nlf4_fc(self.fc4(x))

        return x


class BasicBlockD1(nn.Module):
    expansion = 1

    def __init__(self):
        super(BasicBlockD1, self).__init__()

        densenum = 32
        basenum = 64
        level = 1
        self.bl1 = BasicLayerD(inplanes=level * basenum + densenum * 0)
        self.bl2 = BasicLayerD(inplanes=level * basenum + densenum * 1)
        self.bl3 = BasicLayerD(inplanes=level * basenum + densenum * 2)
        self.bl4 = BasicLayerD(inplanes=level * basenum + densenum * 3)
        self.bl5 = BasicLayerD(inplanes=level * basenum + densenum * 4)
        self.bl6 = BasicLayerD(inplanes=level * basenum + densenum * 5)

        self.nlfT = nn.PReLU()
        self.nlfT2 = nn.PReLU()
        self.convT = hg.Conv2d(level * basenum + densenum * 6, 128, 1, 1)
        self.convT2 = hg.Conv2d(128, 128, 1, 2)

    def forward(self, x):
        residual = x

        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)
        x = self.bl4(x)
        x = self.bl5(x)
        x = self.bl6(x)
        # print("D: ",x.shape)
        x = self.nlfT(self.convT(x))
        x = self.nlfT2(self.convT2(x))
        return x


class BasicBlockD2(nn.Module):
    expansion = 1

    def __init__(self):
        super(BasicBlockD2, self).__init__()

        densenum = 32
        basenum = 128
        level = 1

        self.bl1 = BasicLayerD(inplanes=level * basenum + densenum * 0)
        self.bl2 = BasicLayerD(inplanes=level * basenum + densenum * 1)
        self.bl3 = BasicLayerD(inplanes=level * basenum + densenum * 2)
        self.bl4 = BasicLayerD(inplanes=level * basenum + densenum * 3)
        self.bl5 = BasicLayerD(inplanes=level * basenum + densenum * 4)
        self.bl6 = BasicLayerD(inplanes=level * basenum + densenum * 5)
        self.bl7 = BasicLayerD(inplanes=level * basenum + densenum * 6)
        self.bl8 = BasicLayerD(inplanes=level * basenum + densenum * 7)
        self.bl9 = BasicLayerD(inplanes=level * basenum + densenum * 8)
        self.bl10 = BasicLayerD(inplanes=level * basenum + densenum * 9)
        self.bl11 = BasicLayerD(inplanes=level * basenum + densenum * 10)
        self.bl12 = BasicLayerD(inplanes=level * basenum + densenum * 11)

        self.nlfT = nn.PReLU()
        self.nlfT2 = nn.PReLU()
        self.convT = hg.Conv2d(level * basenum + densenum * 12, 256, 1, 1)
        self.convT2 = hg.Conv2d(256, 256, 1, 2)

    def forward(self, x):
        residual = x

        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)
        x = self.bl4(x)
        x = self.bl5(x)
        x = self.bl6(x)
        x = self.bl7(x)
        x = self.bl8(x)
        x = self.bl9(x)
        x = self.bl10(x)
        x = self.bl11(x)
        x = self.bl12(x)

        x = self.nlfT(self.convT(x))
        x = self.nlfT2(self.convT2(x))
        return x


class BasicBlockD3(nn.Module):
    expansion = 1

    def __init__(self):
        super(BasicBlockD3, self).__init__()
        densenum = 32
        basenum = 256
        level = 1

        self.bl1 = BasicLayerD(inplanes=level * basenum + densenum * 0)
        self.bl2 = BasicLayerD(inplanes=level * basenum + densenum * 1)
        self.bl3 = BasicLayerD(inplanes=level * basenum + densenum * 2)
        self.bl4 = BasicLayerD(inplanes=level * basenum + densenum * 3)
        self.bl5 = BasicLayerD(inplanes=level * basenum + densenum * 4)
        self.bl6 = BasicLayerD(inplanes=level * basenum + densenum * 5)
        self.bl7 = BasicLayerD(inplanes=level * basenum + densenum * 6)
        self.bl8 = BasicLayerD(inplanes=level * basenum + densenum * 7)
        self.bl9 = BasicLayerD(inplanes=level * basenum + densenum * 8)
        self.bl10 = BasicLayerD(inplanes=level * basenum + densenum * 9)
        self.bl11 = BasicLayerD(inplanes=level * basenum + densenum * 10)
        self.bl12 = BasicLayerD(inplanes=level * basenum + densenum * 11)
        self.bl13 = BasicLayerD(inplanes=level * basenum + densenum * 12)
        self.bl14 = BasicLayerD(inplanes=level * basenum + densenum * 13)
        self.bl15 = BasicLayerD(inplanes=level * basenum + densenum * 14)
        self.bl16 = BasicLayerD(inplanes=level * basenum + densenum * 15)
        self.bl17 = BasicLayerD(inplanes=level * basenum + densenum * 16)
        self.bl18 = BasicLayerD(inplanes=level * basenum + densenum * 17)
        self.bl19 = BasicLayerD(inplanes=level * basenum + densenum * 18)
        self.bl20 = BasicLayerD(inplanes=level * basenum + densenum * 19)
        self.bl21 = BasicLayerD(inplanes=level * basenum + densenum * 20)
        self.bl22 = BasicLayerD(inplanes=level * basenum + densenum * 21)
        self.bl23 = BasicLayerD(inplanes=level * basenum + densenum * 22)
        self.bl24 = BasicLayerD(inplanes=level * basenum + densenum * 23)

        self.nlfT = nn.PReLU()
        self.nlfT2 = nn.PReLU()
        self.convT = hg.Conv2d(level * basenum + densenum * 24, 512, 1, 1)
        self.convT2 = hg.Conv2d(512, 512, 1, 2)

    def forward(self, x):
        residual = x

        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)
        x = self.bl4(x)
        x = self.bl5(x)
        x = self.bl6(x)
        x = self.bl7(x)
        x = self.bl8(x)
        x = self.bl9(x)
        x = self.bl10(x)
        x = self.bl11(x)
        x = self.bl12(x)
        x = self.bl13(x)
        x = self.bl14(x)
        x = self.bl15(x)
        x = self.bl16(x)
        x = self.bl17(x)
        x = self.bl18(x)
        x = self.bl19(x)
        x = self.bl20(x)
        x = self.bl21(x)
        x = self.bl22(x)
        x = self.bl23(x)
        x = self.bl24(x)

        x = self.nlfT(self.convT(x))
        x = self.nlfT2(self.convT2(x))
        return x


class BasicLayerD(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, downsample=None, nonlinear_function=None):
        super(BasicLayerD, self).__init__()

        self.conv1 = hg.Conv2d(inplanes, 128, 1)  # conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.PReLU()
        self.conv2 = hg.Conv2d(128, 32, 3, stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        residual = x
        # print("A: ",x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # print("B: ",out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = concat_blocks(out, residual)
        return out


def concat_blocks(x, concatx):
    return torch.cat([x, concatx], dim=1)


class HexNetDeep_dense1_it(nn.Module):
    def __init__(self):
        super(HexNetDeep_dense1_it, self).__init__()
        # size before x.view / batch size
        input_channel = 4

        self.cinput = hg.Conv2d(input_channel, 64, 1, 1)

        self.nlf1 = nn.PReLU()
        self.nlf2 = nn.PReLU()
        self.nlf3 = nn.PReLU()
        self.nlf4 = nn.PReLU()
        self.nlf5 = nn.PReLU()

        self.block1 = BasicBlockD1()
        self.block2 = BasicBlockD2()
        self.block3 = BasicBlockD3()

        self.endpool = nn.AvgPool2d(5)

        self.size2 = 512

        self.fc3 = nn.Linear(self.size2, self.size2)
        self.fc4 = nn.Linear(self.size2, 2)

        self.nlf3_fc = nn.PReLU()
        self.nlf4_fc = nn.PReLU()

    def forward(self, x, vmax, vmin):
        inbs = x.shape[0]
        # vmax = 0
        # vmin = 0 ,vmax,vmin
        
        x = self.cinput(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # print("L: ",x.shape)
        x = self.endpool(x)
        # print("L3: ",x.shape)
        x = x.view(-1, self.size2)
        x = self.nlf3_fc(self.fc3(x))
        x = self.nlf4_fc(self.fc4(x))

        return x

