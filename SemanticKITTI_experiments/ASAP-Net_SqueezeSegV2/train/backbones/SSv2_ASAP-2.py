#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule


class Fire(nn.Module):
  def __init__(self, inplanes, squeeze_planes,
               expand1x1_planes, expand3x3_planes, bn_d=0.1):
    super(Fire, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)
    self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

  def forward(self, x):
    x = self.activation(self.squeeze_bn(self.squeeze(x)))
    return torch.cat([
        self.activation(self.expand1x1_bn(self.expand1x1(x))),
        self.activation(self.expand3x3_bn(self.expand3x3(x)))
    ], 1)


class CAM(nn.Module):

  def __init__(self, inplanes, bn_d=0.1):
    super(CAM, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.pool = nn.MaxPool2d(7, 1, 3)
    self.squeeze = nn.Conv2d(inplanes, inplanes // 16,
                             kernel_size=1, stride=1)
    self.squeeze_bn = nn.BatchNorm2d(inplanes // 16, momentum=self.bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.unsqueeze = nn.Conv2d(inplanes // 16, inplanes,
                               kernel_size=1, stride=1)
    self.unsqueeze_bn = nn.BatchNorm2d(inplanes, momentum=self.bn_d)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # 7x7 pooling
    y = self.pool(x)
    # squeezing and relu
    y = self.relu(self.squeeze_bn(self.squeeze(y)))
    # unsqueezing
    y = self.sigmoid(self.unsqueeze_bn(self.unsqueeze(y)))
    # attention
    return y * x

# ******************************************************************************


MLP1 = [256, 256, 384+192]
FP_MLP1 = [384 + 384 + 256, 256]
RADIUS1 = 0.5
NPOINT1 = 4096
NSAMPLE1 = 16

MLP2 = [384, 384, 512+256]
FP_MLP2 = [512 + 384, 384]
RADIUS2 = 1.0
NPOINT2 = 1024
NSAMPLE2 = 32


class Backbone(nn.Module):
  """
     Class for Squeezeseg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params):
    # Call the super constructor
    super(Backbone, self).__init__()
    print("Using SqueezeNet Backbone")
    self.use_range = params["input_depth"]["range"]
    self.use_xyz = params["input_depth"]["xyz"]
    self.use_remission = params["input_depth"]["remission"]
    self.bn_d = params["bn_d"]
    self.drop_prob = params["dropout"]
    self.OS = params["OS"]

    # input depth calc
    self.input_depth = 0
    self.input_idxs = []
    if self.use_range:
      self.input_depth += 1
      self.input_idxs.append(0)
    if self.use_xyz:
      self.input_depth += 3
      self.input_idxs.extend([1, 2, 3])
    if self.use_remission:
      self.input_depth += 1
      self.input_idxs.append(4)
    print("Depth of backbone input = ", self.input_depth)

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    # make the new stride
    if self.OS > current_os:
      print("Can't do OS, ", self.OS,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == self.OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)

    # encoder
    self.conv1a = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=3,
                                          stride=[1, self.strides[0]],
                                          padding=1),
                                nn.BatchNorm2d(64, momentum=self.bn_d),
                                nn.ReLU(inplace=True),
                                CAM(64, bn_d=self.bn_d))
    self.conv1b = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=1,
                                          stride=1, padding=0),
                                nn.BatchNorm2d(64, momentum=self.bn_d))

    self.fire23 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[1]],
                                             padding=1),
                                Fire(64, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d),
                                Fire(128, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d))
    self.fire45 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[2]],
                                             padding=1),
                                Fire(128, 32, 128, 128, bn_d=self.bn_d),
                                Fire(256, 32, 128, 128, bn_d=self.bn_d))

    self.tem_sa1 = PointnetSAModule(mlp=MLP1, npoint=NPOINT1, nsample=NSAMPLE1, radius=RADIUS1)
    self.fuse_layer1_atten = nn.Conv1d(MLP1[-1]*2, 2, kernel_size=1, stride=1)
    self.fuse_layer1 = nn.Conv1d(MLP1[-1], 384, kernel_size=1, stride=1)
    self.tem_fp1 = PointnetFPModule(mlp=FP_MLP1)

    self.tem_sa2 = PointnetSAModule(mlp=MLP2, npoint=NPOINT2, nsample=NSAMPLE2, radius=RADIUS2)
    self.fuse_layer2_atten = nn.Conv1d(MLP2[-1]*2, 2, kernel_size=1, stride=1)
    self.fuse_layer2 = nn.Conv1d(MLP2[-1], 512, kernel_size=1, stride=1)
    self.tem_fp2 = PointnetFPModule(mlp=FP_MLP2)

    self.fire6789 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                               stride=[1, self.strides[3]],
                                               padding=1),
                                  Fire(256, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 64, 256, 256, bn_d=self.bn_d),
                                  Fire(512, 64, 256, 256, bn_d=self.bn_d))

    # output
    self.dropout = nn.Dropout2d(self.drop_prob)
    self.tem_dropout = nn.Dropout(self.drop_prob)

    # last channels
    self.last_channels = 512

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x):
    # filter input
    x = x[:, :, self.input_idxs]
    x = x.transpose(0, 1)
    frame_num = x.shape[0]
    feat_list = []
    skips_list = []

    for frame_idx in range(frame_num):
        cur_x = x[frame_idx]
        cur_xyz = cur_x[:, 1:4, ...]
        b, c, h, w = cur_xyz.shape
        cur_xyz = cur_xyz.reshape(b, c, -1).transpose(1, 2)

        # run cnn
        # store for skip connections
        skips = {}
        os = 1

        # encoder
        skip_in = self.conv1b(cur_x)
        cur_x = self.conv1a(cur_x)
        # first skip done manually
        skips[1] = skip_in.detach()
        os *= 2

        cur_x, skips, os = self.run_layer(cur_x, self.fire23, skips, os)
        cur_x, skips, os = self.run_layer(cur_x, self.dropout, skips, os)
        cur_x, skips, os = self.run_layer(cur_x, self.fire45, skips, os)

        cur_xyz_l0 = self.xyz_max_layer(cur_xyz, 8)
        b, c, h, w = cur_x.shape
        cur_feat_l0 = cur_x.reshape(b, c, -1).contiguous()

        if frame_idx == 0:
            cur_xyz_l1, cur_prim_feat_l1, sampled_feat_l1 = self.tem_sa1(cur_xyz_l0.contiguous(), cur_feat_l0)
            cur_feat_l1 = self.fuse_layer1(cur_prim_feat_l1)
            cur_xyz_l2, cur_prim_feat_l2, sampled_feat_l2 = self.tem_sa2(cur_xyz_l1, cur_feat_l1)
            cur_feat_l2 = self.fuse_layer2(cur_prim_feat_l2)

            pre_prim_feat_l1 = cur_prim_feat_l1
            pre_prim_feat_l2 = cur_prim_feat_l2
            center_xyz_l1 = cur_xyz_l1
            center_xyz_l2 = cur_xyz_l2
        else:
            # first ASAP layer
            expand_xyz_l0 = torch.cat([cur_xyz_l0, center_xyz_l1], dim=1).contiguous()
            expand_feat_l0 = torch.cat([cur_feat_l0, sampled_feat_l1], dim=-1).contiguous()
            cur_xyz_l1, cur_prim_feat_l1, _ = self.tem_sa1(expand_xyz_l0, expand_feat_l0, center_xyz_l1)
            fuse_feat_l1 = torch.cat([pre_prim_feat_l1, cur_prim_feat_l1], dim=1)
            fuse_feat_l1 = self.tem_dropout(fuse_feat_l1)
            l1_atten = F.softmax(self.fuse_layer1_atten(fuse_feat_l1), dim=1)
            cur_prim_feat_l1 = l1_atten[:, 0, ...].unsqueeze(1) * pre_prim_feat_l1 + \
                          l1_atten[:, 1, ...].unsqueeze(1) * cur_prim_feat_l1
            cur_feat_l1 = self.fuse_layer1(cur_prim_feat_l1) 
            cur_feat_l1 = self.tem_dropout(cur_feat_l1)

            # second ASAP layer
            expand_xyz_l1 = torch.cat([cur_xyz_l1, center_xyz_l2], dim=1).contiguous()
            expand_feat_l1 = torch.cat([cur_feat_l1, sampled_feat_l2], dim=-1).contiguous()
            cur_xyz_l2, cur_prim_feat_l2, _ = self.tem_sa2(expand_xyz_l1, expand_feat_l1, center_xyz_l2)
            fuse_feat_l2 = torch.cat([pre_prim_feat_l2, cur_prim_feat_l2], dim=1)
            fuse_feat_l2 = self.tem_dropout(fuse_feat_l2)
            l2_atten = F.softmax(self.fuse_layer2_atten(fuse_feat_l2), dim=1)
            cur_prim_feat_l2 = l2_atten[:, 0, ...].unsqueeze(1) * pre_prim_feat_l2 + \
                          l2_atten[:, 1, ...].unsqueeze(1) * cur_prim_feat_l2
            cur_feat_l2 = self.fuse_layer2(cur_prim_feat_l2) 
            cur_feat_l2 = self.tem_dropout(cur_feat_l2)
            
            pre_prim_feat_l1 = cur_prim_feat_l1
            pre_prim_feat_l2 = cur_prim_feat_l2

        cur_x = self.tem_fp2(cur_xyz_l1, cur_xyz_l2, cur_feat_l1, cur_feat_l2)
        feat_fp1 = self.tem_dropout(torch.cat([cur_x, cur_feat_l1], dim=1))
        cur_x = self.tem_fp1(cur_xyz_l0, cur_xyz_l1, cur_feat_l0, feat_fp1)
        cur_x = cur_x.reshape(b, c, h, w)

        cur_x, skips, os = self.run_layer(cur_x, self.dropout, skips, os)
        cur_x, skips, os = self.run_layer(cur_x, self.fire6789, skips, os)
        cur_x, skips, os = self.run_layer(cur_x, self.dropout, skips, os)

        feat_list.append(cur_x.unsqueeze(1))
        skips_list.append(skips)

    out = torch.cat(feat_list, dim=1)

    return out, skips_list

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth

  def xyz_max_layer(self, _input, rate):
    b, n, c = _input.shape
    out = _input.reshape([b, n // rate, int(rate), c])
    out, _ = torch.max(out, dim=-2)
    return out

  def xyz_mid_layer(self, _input, rate):
    b, n, c = _input.shape
    out = _input.reshape([b, n // rate, int(rate), c])
    out, _ = torch.median(out, dim=-2)
    return out


