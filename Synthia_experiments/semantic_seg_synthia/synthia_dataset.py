'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import copy
import psutil
import random
from pyquaternion import Quaternion
import class_mapping

class SegDataset():
    def __init__(self, root='processed_pc', \
            filelist_name='data_prep/train_raw.txt', \
            labelweight_filename = 'data_prep/labelweights.npz', \
            npoints = 16384, num_frames=1, train=True):
        self.npoints = npoints
        self.train = train
        self.root = root
        self.num_frames = num_frames

        self.labelweights = np.load(labelweight_filename)['labelweights']

        filenames = []
        raw_txt_file = open(filelist_name, 'r')
        l = raw_txt_file.readline()
        while len(l) > 0:
            l = l.split(' ')[0]
            l = l.split('/')
            sequence_name = l[0]
            frame_id = int(l[-1].split('.')[0])

            filenames.append([sequence_name, frame_id])
            l = raw_txt_file.readline()

        filenames.sort()
        self.filenames = filenames

        self.start_points = []
        if self.train:
            current_seq = self.filenames[0][0]
            frame_cnt = 0
            for i in range(self.filenames.__len__()):
                if i == self.filenames.__len__() - 1:
                    frame_cnt += 1
                    random_idx = random.sample(range(frame_cnt - self.num_frames + 1), 
                                                int((frame_cnt - self.num_frames) / self.num_frames))
                    for j in range(random_idx.__len__()):
                        self.start_points.append([current_seq, random_idx[j]])
                
                elif self.filenames[i][0] == current_seq:
                    frame_cnt += 1
                
                else:
                    random_idx = random.sample(range(frame_cnt - self.num_frames + 1), 
                                                int((frame_cnt - self.num_frames) / self.num_frames))

                    for j in range(random_idx.__len__()):
                        self.start_points.append([current_seq, random_idx[j]])
                    
                    current_seq = self.filenames[i][0]
                    frame_cnt = 1
        else:
            current_seq = self.filenames[0][0]
            i = 0
            while i + self.num_frames < self.filenames.__len__():
                if current_seq == self.filenames[i + self.num_frames - 1][0]:
                    self.start_points.append([current_seq, self.filenames[i][1]])
                    i += self.num_frames
                    current_seq = self.filenames[i][0]
                else:
                    flag = 0
                    for j in range(1, self.num_frames):
                        if self.filenames[i + self.num_frames - 1 - j][0] == current_seq:
                            end = i + self.num_frames - 1 - j
                            flag = 1
                            break
                    if not flag:
                        print('current_seq: ', current_seq)
                        for j in range(self.num_frames):
                            print(self.filenames[i + j])
                    self.start_points.append([current_seq, self.filenames[end - self.num_frames + 1][1]])
                    i = end + 1
                    current_seq = self.filenames[i][0]
            
            if self.filenames[-self.num_frames] not in self.start_points:
                self.start_points.append(self.filenames[-self.num_frames])

        ##### debug
        # self.filenames = [f for f in self.filenames if 'SYNTHIA-SEQS-01-DAWN' in f[0]]

        self.cache = {}
        self.cache_mem_usage = 0.95

    def read_data(self, sequence_name, frame_id):
        if sequence_name in self.cache:
            if frame_id in self.cache[sequence_name]:
                pc, rgb, semantic, center = self.cache[sequence_name][frame_id]
                return pc, rgb, semantic, center

        fn = os.path.join(self.root, sequence_name + '-' + str(frame_id).zfill(6) + '.npz')
        data = np.load(fn)

        pc = data['pc']
        rgb = data['rgb']
        semantic = data['semantic']
        center = data['center']
        semantic = semantic.astype('uint8')

        mem = psutil.virtual_memory()
        if (mem.used / mem.total) < self.cache_mem_usage:
            if sequence_name not in self.cache:
                self.cache[sequence_name] = {}
            self.cache[sequence_name][frame_id] = (pc, rgb, semantic, center)
        return pc, rgb, semantic, center

    def data_reset(self):
        if self.train:
            self.start_points = []
            current_seq = self.filenames[0][0]
            frame_cnt = 0
            for i in range(self.filenames.__len__()):
                if i == self.filenames.__len__() - 1:
                    frame_cnt += 1
                    random_idx = random.sample(range(frame_cnt - self.num_frames + 1), 
                                                int((frame_cnt - self.num_frames) / self.num_frames))
                    
                    for j in range(random_idx.__len__()):
                        self.start_points.append([current_seq, random_idx[j]])
                elif self.filenames[i][0] == current_seq:
                    frame_cnt += 1
                else:
                    random_idx = random.sample(range(frame_cnt - self.num_frames + 1), 
                                                int((frame_cnt - self.num_frames) / self.num_frames))
                    
                    for j in range(random_idx.__len__()):
                        self.start_points.append([current_seq, random_idx[j]])
                    current_seq = self.filenames[i][0]
                    frame_cnt = 1

    def half_crop_w_context(self, half, context, pc, rgb, semantic, center):
        all_idx = np.arange(pc.shape[0])
        if half == 0:
            sample_idx_half_w_context = all_idx[pc[:, 2] > (center[2] - context)]
            pc_half_w_context = pc[sample_idx_half_w_context]
            loss_mask = pc_half_w_context[:, 2] > center[2]
        else:
            sample_idx_half_w_context = all_idx[pc[:, 2] < (center[2] + context)]
            pc_half_w_context = pc[sample_idx_half_w_context]
            loss_mask = pc_half_w_context[:, 2] < center[2]

        valid_pred_idx_in_full = sample_idx_half_w_context

        pc_half_w_context = pc[sample_idx_half_w_context]
        rgb_half_w_context = rgb[sample_idx_half_w_context]
        semantic_half_w_context = semantic[sample_idx_half_w_context]
        return pc_half_w_context, rgb_half_w_context, semantic_half_w_context, \
                loss_mask, valid_pred_idx_in_full

    def augment(self, pc, center, flip, scale, rot_axis, rot_angle):
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center

        pc = (pc - center) * scale + center

        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center
        return pc

    def mask_and_label_conversion(self, semantic, loss_mask):
        semantic = semantic.astype('int32')
        label = class_mapping.index_to_label_vec_func(semantic)
        loss_mask = (label != 12) * loss_mask
        label[label == 12] = 0
        
        return label, loss_mask

    def choice_to_num_points(self, pc, rgb, label, loss_mask, valid_pred_idx_in_full):

        # shuffle idx to change point order (change FPS behavior)
        idx = np.arange(pc.shape[0])
        choice_num = self.npoints
        if pc.shape[0] > choice_num:
            shuffle_idx = np.random.choice(idx, choice_num, replace=False)
        else:
            shuffle_idx = np.concatenate([np.random.choice(idx, choice_num -  idx.shape[0]), \
                    np.arange(idx.shape[0])])

        pc = pc[shuffle_idx]
        rgb = rgb[shuffle_idx]
        label = label[shuffle_idx]
        loss_mask = loss_mask[shuffle_idx]
        if valid_pred_idx_in_full is not None:
            valid_pred_idx_in_full = valid_pred_idx_in_full[shuffle_idx]

        return pc, rgb, label, loss_mask, valid_pred_idx_in_full

    def get(self, index, half=0, context=1.):
        
        slice_invalid = True
        while slice_invalid:
            sequence_name, start_frame = self.start_points[index]
            cnt = 0
            for offset in range(self.num_frames):
                frame_id = start_frame + offset
                fn = os.path.join(self.root, sequence_name + '-' + str(frame_id).zfill(6) + '.npz')
                if os.path.exists(fn):
                    cnt += 1
            if cnt == self.num_frames:
                slice_invalid = False
            else:
                index = (index + 1) % self.start_points.__len__()
        
        pc_list = []
        rgb_list = []
        label_list = []
        loss_mask_list = []
        valid_idx_list = []
        path_list = []

        flip = np.random.uniform(0, 1) > 0.5
        scale = np.random.uniform(0.8, 1.2)
        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * 2)

        for offset in range(self.num_frames):
            frame_idx = start_frame + offset
            pc, rgb, semantic, center = self.read_data(sequence_name, frame_idx)
            path_list.append((sequence_name, frame_idx))
            
            valid_pred_idx_in_full = None
            loss_mask = pc[:, 0] != 0
            
            label, loss_mask = self.mask_and_label_conversion(semantic, loss_mask)
            
            pc, rgb, label, loss_mask, valid_pred_idx_in_full = \
                self.choice_to_num_points(pc, rgb, label, loss_mask, valid_pred_idx_in_full)
            
            if self.train:
                pc = self.augment(pc, center, flip, scale, rot_axis, rot_angle)
            
            pc_list.append(np.expand_dims(pc, 0))
            rgb_list.append(np.expand_dims(rgb, 0))
            label_list.append(np.expand_dims(label, 0))
            loss_mask_list.append(np.expand_dims(loss_mask, 0))
            valid_idx_list.append(np.expand_dims(valid_pred_idx_in_full, 0))
        
        pcs = np.concatenate(pc_list, axis=0)
        rgbs = np.concatenate(rgb_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        loss_masks = np.concatenate(loss_mask_list, axis=0)
        valid_idxes = np.concatenate(valid_idx_list, axis=0)

        if self.train:
            labelweights = 1/np.log(1.2 + self.labelweights)
            labelweights = labelweights / labelweights.min()
        else:
            labelweights = np.ones_like(self.labelweights)

        return pcs, rgbs, labels, labelweights, loss_masks, valid_idxes, path_list

    def __len__(self):
        return len(self.start_points)


if __name__ == '__main__':
    import mayavi.mlab as mlab
    import class_mapping
    NUM_POINT = 8192
    num_frames = 2
    d = SegDataset(root='processed_pc', npoints=NUM_POINT, train=True, num_frames=num_frames)
    print(len(d))
    import time
    tic = time.time()
    point_size = 0.2
    for idx in range(200, len(d)):
        for half in [0, 1]:

            batch_data = np.zeros((NUM_POINT * num_frames, 3 + 3))
            batch_label = np.zeros((NUM_POINT * num_frames), dtype='int32')
            batch_mask = np.zeros((NUM_POINT * num_frames), dtype=np.bool)

            pc, rgb, label, labelweights, loss_mask, valid_pred_idx_in_full, path_list = d.get(idx, half)

            batch_data[:, :3] = pc
            batch_data[:, 3:] = rgb
            batch_label = label
            batch_mask = loss_mask

            batch_labelweights = labelweights[batch_label]

            batch_data = batch_data[:NUM_POINT]
            batch_label = batch_label[:NUM_POINT]
            batch_mask = batch_mask[:NUM_POINT]
            batch_labelweights = batch_labelweights[:NUM_POINT]

            mlab.figure(bgcolor=(1,1,1))

            pc_valid = batch_data[:, :3][batch_mask]
            rgb_valid = batch_data[:, 3:][batch_mask]
            label_valid = batch_label[batch_mask]
            for i in range(12):
                pc_sem = pc_valid[label_valid == i]
                color = class_mapping.index_to_color[class_mapping.label_to_index[i]]

                mlab.points3d(pc_sem[:,0], pc_sem[:,1], pc_sem[:,2], scale_factor=point_size, color=(color[0]/255,color[1]/255,color[2]/255))

            pc_non_valid = batch_data[:, :3][np.logical_not(batch_mask)]
            mlab.points3d(pc_non_valid[:,0], pc_non_valid[:,1], pc_non_valid[:,2], scale_factor=point_size, color=(0, 0, 0))

            f = open('view.pts', 'w')
            for i in range(batch_data.shape[0]):
                p = batch_data[i, :3]
                color = 2 * batch_data[i, 3:] - 1
                ##### write color
                f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], color[0], color[1], color[2]))


            input()

    print(time.time() - tic)


