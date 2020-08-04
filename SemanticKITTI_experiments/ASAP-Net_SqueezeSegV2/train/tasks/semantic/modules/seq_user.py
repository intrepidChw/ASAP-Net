#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np

from tasks.semantic.modules.seq_segmentator import *
from tasks.semantic.postproc.KNN import KNN


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/seq_parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      mode='test',
                                      frame_num=self.ARCH["test"]["frame_num"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=self.ARCH["test"]["batch_size"],
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    # do train set
    # self.infer_subset(loader=self.parser.get_train_set(),
    #                   to_orig_fn=self.parser.to_original)
    
    # do valid set
    # self.infer_subset(loader=self.parser.get_valid_set(),
    #                   to_orig_fn=self.parser.to_original)
    # do test set
    self.infer_subset(loader=self.parser.get_test_set(),
                      to_orig_fn=self.parser.to_original)

    print('Finished Infering')

    return

  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        
        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        # compute output
        proj_output = self.model(proj_in, proj_mask)
        proj_argmax = proj_output.argmax(dim=2)

        if self.post:
          # knn postproc
          # unproj_argmax = self.post(proj_range,
          #                           unproj_range,
          #                           proj_argmax,
          #                           p_x,
          #                           p_y)
          raise NotImplementedError()
        else:
          # put in original pointcloud using indexes
          for batch_idx in range(proj_in.shape[0]):
              for frame_idx in range(proj_in.shape[1]):
                  # print('npoints:', npoints[frame_idx][batch_idx])
                  # print('p_x:', p_x.shape)
                  cur_p_x = p_x[batch_idx, frame_idx, 0:npoints[frame_idx][batch_idx]]
                  cur_p_y = p_y[batch_idx, frame_idx, 0:npoints[frame_idx][batch_idx]]
                  cur_unproj_argmax = proj_argmax[batch_idx, frame_idx, cur_p_y, cur_p_x]

                  # measure elapsed time
                  if torch.cuda.is_available():
                      torch.cuda.synchronize()

                  print("Infered seq", path_seq[frame_idx][batch_idx], "scan", path_name[frame_idx][batch_idx],
                        "in", time.time() - end, "sec")
                  end = time.time()

                  # save scan
                  # get the first scan in batch and project scan
                  pred_np = cur_unproj_argmax.cpu().numpy()
                  pred_np = pred_np.reshape((-1)).astype(np.int32)

                  # map to original label
                  pred_np = to_orig_fn(pred_np)

                  # save scan
                  pred_dir = os.path.join(self.logdir, "sequences",
                                      path_seq[frame_idx][batch_idx], "predictions")
                  os.makedirs(pred_dir, exist_ok=True)
                  path = os.path.join(self.logdir, "sequences",
                                      path_seq[frame_idx][batch_idx], "predictions", path_name[frame_idx][batch_idx])
                  
                  pred_np.tofile(path)
          # unproj_argmax = proj_argmax[p_y, p_x]


