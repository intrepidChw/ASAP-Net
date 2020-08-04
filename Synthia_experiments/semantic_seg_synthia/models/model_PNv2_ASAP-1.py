"""
    Compared with model_baseline, do not use correlation output for skip link
    Compared to model_baseline_fixed, added return values to test whether nsample is set reasonably.
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../..'))
sys.path.append(os.path.join(BASE_DIR, '../../tf_ops/sampling'))
import tf_util
from net_utils import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, num_point, num_frames):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frames, num_point, 3 + 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_frames, num_point))
    labelweights_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frames, num_point))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_frames, num_point))
    return pointclouds_pl, labels_pl, labelweights_pl, masks_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    # batch_size = point_cloud.get_shape()[0].value
    num_frame = point_cloud.get_shape()[1].value
    # num_point = point_cloud.get_shape()[2].value

    out_list = []

    for frame_idx in range(num_frame):
        cur_point_cloud = point_cloud[:, frame_idx, :, :]

        l0_xyz = cur_point_cloud[:, :, 0:3]
        l0_points = cur_point_cloud[:, :, 3:]

        RADIUS1 = 1.0
        RADIUS2 = RADIUS1 * 2
        RADIUS3 = RADIUS1 * 4
        RADIUS4 = RADIUS1 * 8
        RADIUS_TEM = 4.0

        l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points, new_xyz=None, npoint=2048, radius=RADIUS1, nsample=32, mlp=[32,32,128], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, new_xyz=None, npoint=512, radius=RADIUS2, nsample=32, mlp=[64,64,256], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        
        # Temporal Unit
        if frame_idx == 0:
            tem_xyz_l1, tem_points_l1, idx_tuple = pointnet_sa_module(l2_xyz, l2_points, new_xyz=None, npoint=128, radius=RADIUS_TEM, nsample=32, mlp=[128,128,768], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='tem_layer1')
            # print('idx:', idx_tuple[1].shape)
            sampled_idx = idx_tuple[1]
            batch_idx = tf.expand_dims(tf.expand_dims(tf.range(sampled_idx.shape[0], dtype=tf.int32), axis=1), axis=2)
            batch_idx = tf.tile(batch_idx, [1, sampled_idx.shape[1], 1])
            sampled_idx = tf.expand_dims(sampled_idx, axis=-1)
            sampled_idx = tf.concat([batch_idx, sampled_idx], axis=-1)
            center_xyz = tem_xyz_l1
            center_feat = tf.gather_nd(l2_points, sampled_idx)
            # print('center feat:', center_feat.shape)
            pre_tem_points_l1 = tem_points_l1
            l3_points = tf_util.conv1d(tem_points_l1, 512, 1,
                                        padding='VALID', stride=1,
                                        bn=True, is_training=is_training,
                                        scope='tem_fuse_layer1', bn_decay=bn_decay,
                                        data_format='NHWC')
        else:
            l2_xyz = tf.concat([l2_xyz, center_xyz], axis=-2)
            l2_points = tf.concat([l2_points, center_feat], axis=-2)
            tem_xyz_l1, tem_points_l1, _ = pointnet_sa_module(l2_xyz, l2_points, new_xyz=center_xyz, npoint=128, radius=RADIUS_TEM, nsample=32, mlp=[128,128,768], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='tem_layer1')
            cat_points_l1 = tf.concat([pre_tem_points_l1, tem_points_l1], axis=-1)
            atten_l1 = tf_util.conv1d(cat_points_l1, 2, 1,
                                        padding='VALID', stride=1,
                                        bn=True, is_training=is_training,
                                        scope='atten_layer1', bn_decay=bn_decay,
                                        data_format='NHWC')
            atten_l1 = tf.nn.softmax(atten_l1, axis=-1)
            tem_points_l1 = pre_tem_points_l1 * tf.expand_dims(atten_l1[..., 0], axis=-1) + tem_points_l1 * tf.expand_dims(atten_l1[..., 1], axis=-1)
            
            l3_points = tf_util.conv1d(tem_points_l1, 512, 1,
                                        padding='VALID', stride=1,
                                        bn=True, is_training=is_training,
                                        scope='tem_fuse_layer1', bn_decay=bn_decay,
                                        data_format='NHWC')
            pre_tem_points_l1 = tem_points_l1
        
        # l2_points = pointnet_fp_module(l2_xyz, tem_xyz_l1, l2_points, tem_points_l1, [512,256], is_training, bn_decay, scope='tem_fa_layer1')
        
        # l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, new_xyz=None, npoint=128, radius=RADIUS3, nsample=32, mlp=[128,128,512], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
        l3_xyz = tem_xyz_l1
        # l3_points = tem_points_l1
        l4_xyz, l4_points, _ = pointnet_sa_module(l3_xyz, l3_points, new_xyz=None, npoint=64, radius=RADIUS4, nsample=32, mlp=[256,256,1024], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128], is_training, bn_decay, scope='fa_layer4')

        ##### debug
        net = tf_util.conv1d(l0_points, 12, 1, padding='VALID', activation_fn=None, scope='fc2')
        out_list.append(tf.expand_dims(net, axis=1))
    
    out = tf.concat(out_list, axis=1)
   
    return out, end_points

def get_loss(pred, label, mask, end_points, label_weights):
    """ pred: BxNx3,
        label: BxN,
        mask: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy( labels=label, \
                                                            logits=pred, \
                                                            weights=label_weights, \
                                                            reduction=tf.losses.Reduction.NONE)
    classify_loss = tf.reduce_sum(classify_loss * mask) / (tf.reduce_sum(mask) + 1)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)

    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
