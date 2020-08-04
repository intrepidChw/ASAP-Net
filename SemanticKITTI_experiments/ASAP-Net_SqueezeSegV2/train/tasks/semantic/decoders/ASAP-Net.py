#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class FireUp(nn.Module):

  def __init__(self, inplanes, squeeze_planes,
               expand1x1_planes, expand3x3_planes, bn_d, stride):
    super(FireUp, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.stride = stride
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
    if self.stride == 2:
      self.upconv = nn.ConvTranspose2d(squeeze_planes, squeeze_planes,
                                       kernel_size=[1, 4], stride=[1, 2],
                                       padding=[0, 1])
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)
    self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

  def forward(self, x):
    x = self.activation(self.squeeze_bn(self.squeeze(x)))
    if self.stride == 2:
      x = self.activation(self.upconv(x))
    return torch.cat([
        self.activation(self.expand1x1_bn(self.expand1x1(x))),
        self.activation(self.expand3x3_bn(self.expand3x3(x)))
    ], 1)


# ******************************************************************************

class Decoder(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params, stub_skips, OS=32, feature_depth=512):
    super(Decoder, self).__init__()
    self.backbone_OS = OS
    self.backbone_feature_depth = feature_depth
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Decoder original OS: ", int(current_os))
    # redo strides according to needed stride
    for i, stride in enumerate(self.strides):
      if int(current_os) != self.backbone_OS:
        if stride == 2:
          current_os /= 2
          self.strides[i] = 1
        if int(current_os) == self.backbone_OS:
          break
    print("Decoder new OS: ", int(current_os))
    print("Decoder strides: ", self.strides)

    # decoder
    # decoder
    self.firedec10 = FireUp(self.backbone_feature_depth,
                            64, 128, 128, bn_d=self.bn_d,
                            stride=self.strides[0])
    self.firedec11 = FireUp(256, 32, 64, 64, bn_d=self.bn_d,
                            stride=self.strides[1])
    self.firedec12 = FireUp(128, 16, 32, 32, bn_d=self.bn_d,
                            stride=self.strides[2])
    self.firedec13 = FireUp(64, 16, 32, 32, bn_d=self.bn_d,
                            stride=self.strides[3])

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 64

  def run_layer(self, x, layer, skips, os):
    feats = layer(x)  # up
    if feats.shape[-1] > x.shape[-1]:
      os //= 2  # match skip
      # print('os:', os)
      feats = feats + skips[os].detach()  # add skip
    x = feats
    return x, skips, os

  def forward(self, x, skips):
    # print('os:', os)
    x = x.transpose(0, 1)
    # skips = skips.transpose(0, 1)
    frame_num = x.shape[0]

    out_list = []
    for frame_idx in range(frame_num):
      cur_x = x[frame_idx]
      cur_skips = skips[frame_idx]
      os = self.backbone_OS
      # run layers
      cur_x, cur_skips, os = self.run_layer(cur_x, self.firedec10, cur_skips, os)
      # print('os1:', os)
      cur_x, cur_skips, os = self.run_layer(cur_x, self.firedec11, cur_skips, os)
      # print('os2:', os)
      cur_x, cur_skips, os = self.run_layer(cur_x, self.firedec12, cur_skips, os)
      # print('os3:', os)
      cur_x, cur_skips, os = self.run_layer(cur_x, self.firedec13, cur_skips, os)
      # print('os4:', os)

      cur_x = self.dropout(cur_x)
      out_list.append(cur_x.unsqueeze(1))

    out = torch.cat(out_list, dim=1)
    return out

  def get_last_depth(self):
    return self.last_channels
