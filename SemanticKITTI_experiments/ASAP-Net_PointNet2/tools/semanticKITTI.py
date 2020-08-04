import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from tools.common.laserscan import SemLaserScan, LaserScan


class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               mode,
               frame_num,
               sample_points=45000,   # sampling number of points present in dataset
               max_points=150000,     # max number of points present in dataset
               gt=True):            # send ground truth?

    # save deats
    self.root = os.path.join(root, "sequences")
    self.image_set = os.path.join(self.root, 'ImageSet2', mode+'.txt')
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.mode = mode
    self.frame_num = frame_num
    self.max_points = max_points
    self.sample_points = sample_points
    self.gt = gt

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    with open(self.image_set, 'r') as f:
        frames = f.readlines()

    for frame in frames:
        frame_split = frame.strip().split('/')
        # if '08' in frame_split[0]:
        self.scan_files.append(os.path.join(self.root, frame_split[0], 'velodyne', frame_split[1]+'.bin'))
        self.label_files.append(os.path.join(self.root, frame_split[0], 'labels', frame_split[1]+'.label'))
    print('scans:', len(self.scan_files))
    print('frame_num:', self.frame_num)
    print('__len__:', self.__len__())

  def data_reset(self):
    self.scan_files = []
    self.label_files = []
    train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    for seq in train_seqs:
      seq_dir = os.path.join(self.root, "%02d" % seq)
      velo_dir = os.path.join(seq_dir, 'velodyne')
      frames = os.listdir(velo_dir)
      frame_num = len(frames)
      for _ in range(frame_num // self.frame_num):
        start_idx = random.randint(0, frame_num - self.frame_num)
        for offset in range(0, self.frame_num):
          frame_idx = start_idx + offset
          self.scan_files.append(os.path.join(self.root, '%02d' % seq, 'velodyne', '%06d' % frame_idx + '.bin'))
          self.label_files.append(os.path.join(self.root, '%02d' % seq, 'labels', '%06d' % frame_idx + '.label'))

    print('Dataset reset!')
    print('scans:', len(self.scan_files))
    print('labels:', len(self.label_files))

  def __getitem__(self, index):
    frame_num = self.frame_num

    npoints_list = []
    
    path_seq_list = []
    path_name_list = []
    
    full_xyz_list = []
    full_range_list = []
    full_remissions_list = []
    full_labels_list = []
    
    selected_xyz_list = []
    selected_range_list = []
    selected_remissions_list = []
    selected_labels_list = []

    for idx in range(index*frame_num, index*frame_num+frame_num):
        # get item in tensor shape
        scan_file = self.scan_files[idx]
        if self.gt:
          label_file = self.label_files[idx]

        # open a semantic laserscan
        if self.gt:
          scan = SemLaserScan(self.color_map,
                              project=True,
                              H=self.sensor_img_H,
                              W=self.sensor_img_W,
                              fov_up=self.sensor_fov_up,
                              fov_down=self.sensor_fov_down)
        else:
          scan = LaserScan(project=True,
                           H=self.sensor_img_H,
                           W=self.sensor_img_W,
                           fov_up=self.sensor_fov_up,
                           fov_down=self.sensor_fov_down)

        # open and obtain scan
        scan.open_scan(scan_file)
        if self.gt:
          scan.open_label(label_file)
          # map unused classes to used classes (also for projection)
          scan.sem_label = self.map(scan.sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        full_n_points = scan.points.shape[0]
        selected_idx = np.random.choice(full_n_points, self.sample_points)
        
        # print('selected_idx:', selected_idx.shape)
        # print('points:', scan.points.shape)
        full_xyz = torch.full((self.max_points, 3), 0.0, dtype=torch.float)
        # selected_xyz = torch.full((self.sample_points, 3), 0.0, dtype=torch.float)
        full_xyz[:full_n_points] = torch.from_numpy(scan.points)
        selected_xyz = torch.from_numpy(scan.points[selected_idx, :])
        
        # full_xyz = torch.from_numpy(full_xyz)
        full_range = torch.full([self.max_points], 0.0, dtype=torch.float)
        # selected_range = torch.full([self.sample_points], 0.0, dtype=torch.float)
        full_range[:full_n_points] = torch.from_numpy(scan.unproj_range)
        selected_range = torch.from_numpy(scan.unproj_range[selected_idx])
        
        full_remissions = torch.full([self.max_points], 0.0, dtype=torch.float)
        full_remissions[:full_n_points] = torch.from_numpy(scan.remissions)
        selected_remissions = torch.from_numpy(scan.remissions[selected_idx])

        if self.gt:
          full_labels = torch.full([self.max_points], 0, dtype=torch.int32)
          full_labels[:full_n_points] = torch.from_numpy(scan.sem_label)
          selected_labels = torch.from_numpy(scan.sem_label[selected_idx])
        else:
          full_labels = []
          selected_labels = []

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        # print("path_norm: ", path_norm)
        # print("path_seq", path_seq)
        # print("path_name", path_name)
        if self.gt:
            full_labels_list.append(full_labels.unsqueeze(0))
            selected_labels_list.append(selected_labels.unsqueeze(0))

        path_seq_list.append(path_seq)
        path_name_list.append(path_name)
        full_range_list.append(full_range.unsqueeze(0))
        full_xyz_list.append(full_xyz.unsqueeze(0))
        full_remissions_list.append(full_remissions.unsqueeze(0))
        npoints_list.append(full_n_points)

        selected_xyz_list.append(selected_xyz.unsqueeze(0))
        selected_range_list.append(selected_range.unsqueeze(0))
        selected_remissions_list.append(selected_remissions.unsqueeze(0))

    if self.gt:
        full_labels_seq = torch.cat(full_labels_list, dim=0)
        selected_labels_seq = torch.cat(selected_labels_list, dim=0)
    else:
        full_labels_seq = []
        selected_labels_seq = []

    # path_seq = torch.cat(path_seq_list)
    full_range_seq = torch.cat(full_range_list, dim=0)
    full_xyz_seq = torch.cat(full_xyz_list, dim=0)
    full_remissions_seq = torch.cat(full_remissions_list, dim=0)

    selected_range_seq = torch.cat(selected_range_list, dim=0)
    selected_xyz_seq = torch.cat(selected_xyz_list, dim=0)
    selected_remissions_seq = torch.cat(selected_remissions_list, dim=0)

    # return
    return full_labels_seq, path_seq_list, path_name_list, full_range_seq, \
        full_xyz_seq, full_remissions_seq, npoints_list, selected_xyz_seq, \
        selected_range_seq, selected_remissions_seq, selected_labels_seq

  def __len__(self):
    return len(self.scan_files) // self.frame_num

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]


