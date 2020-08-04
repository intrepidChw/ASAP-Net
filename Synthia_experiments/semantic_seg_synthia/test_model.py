import argparse
import math
from datetime import datetime
#import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models')) # model
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../data'))

import synthia_dataset
import class_mapping

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', default='processed_pc', help='Dataset dir [default: model_basic]')
parser.add_argument('--model', default='model_basic', help='Model name [default: model_basic]')
parser.add_argument('--model_path', default=None, help='Model path to restore [default: None]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--num_frame', type=int, default=1, help='Number of frames [default: 1]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--command_file', default=None, help='Name of command file [default: None]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--save', type=bool, default=False, help='Whether to save results.')
parser.add_argument('--save_dir', default='results', help='result dir [default: results]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)

BATCH_SIZE = FLAGS.batch_size
DATA = FLAGS.data
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frame
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
COMMAND_FILE = FLAGS.command_file

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
os.system('cp %s %s' % (COMMAND_FILE, LOG_DIR)) # bkp of command file
os.system('cp %s %s' % ('synthia_dataset_tpc.py', LOG_DIR)) # bkp of command file
os.system('cp ../utils/net_utils.py %s' % (LOG_DIR)) # bkp of train procedure

LOG_TEST_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_TEST_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 12

# TEST_DATASET = synthia_dataset_tpc_save.SegDataset(DATA, filelist_name='data_prep/trainval_raw.txt', npoints=NUM_POINT, num_frames=NUM_FRAME, train=False)
TEST_DATASET = synthia_dataset.SegDataset(DATA, filelist_name='data_prep/test_raw.txt', npoints=NUM_POINT, num_frames=NUM_FRAME, train=False)

def log_test_string(out_str):
    LOG_TEST_FOUT.write(out_str+'\n')
    LOG_TEST_FOUT.flush()
    print(out_str)

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def count_params():
    return int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, labelweights_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            # print('pred:', pred.get_shape())
            loss = MODEL.get_loss(pred, labels_pl, masks_pl, end_points, labelweights_pl)
            tf.summary.scalar('loss', loss)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        #sess.run(init, {is_training_pl: True})

        log_test_string('Model parameters: %d' % count_params())

        if (MODEL_PATH is not None) and (MODEL_PATH != 'None'):
            # variables = tf.contrib.framework.get_variables_to_restore()
            # variables_to_resotre = [v for v in variables if 'tem' not in v.name]
            # saver = tf.train.Saver(variables_to_resotre)
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_PATH)
            log_test_string('Model restored.')

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'labelweights_pl': labelweights_pl,
               'masks_pl': masks_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        epoch = int(MODEL_PATH.split('/')[-1].split('.')[-2].split('-')[-1])
        # log_string('---- EPOCH %03d TEST ----'%(epoch))
        eval_one_epoch(sess, ops, test_writer, dataset=TEST_DATASET, epoch_cnt=epoch)
            

def get_batch(dataset, idxs, start_idx, end_idx, half=0):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_FRAME, NUM_POINT, 3 + 3))
    batch_label = np.zeros((bsize, NUM_FRAME, NUM_POINT), dtype='int32')
    batch_mask = np.zeros((bsize, NUM_FRAME, NUM_POINT), dtype=np.bool)
    fns = []

    for i in range(bsize):
        pc, rgb, label, labelweights, loss_mask, _, fn = dataset.get(idxs[i+start_idx], half)

        batch_data[i, :, :, :3] = pc
        batch_data[i, :, :, 3:] = rgb
        batch_label[i] = label
        batch_mask[i] = loss_mask
        fns.append(fn)

    batch_labelweights = labelweights[batch_label]

    return batch_data, batch_label, batch_labelweights, batch_mask, fns

def eval_one_epoch(sess, ops, test_writer, dataset, epoch_cnt):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(dataset))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(dataset)+BATCH_SIZE-1) // BATCH_SIZE

    loss_sum = 0
    total_pred_label_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_class = [0 for _ in range(NUM_CLASSES)]

    log_test_string(str(datetime.now()))
    log_test_string('---- EPOCH %03d EVALUATION ----'%(epoch_cnt))

    batch_data = np.zeros((BATCH_SIZE, NUM_FRAME, NUM_POINT, 3 + 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_FRAME, NUM_POINT))
    batch_mask = np.zeros((BATCH_SIZE, NUM_FRAME, NUM_POINT))
    batch_labelweights = np.zeros((BATCH_SIZE, NUM_FRAME, NUM_POINT))
    for batch_idx in range(num_batches):
        
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(dataset), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        for half in [0, 1]:
            cur_batch_data, cur_batch_label, cur_batch_labelweights, cur_batch_mask, fns = \
                    get_batch(dataset, test_idxs, start_idx, end_idx, half)
            if cur_batch_size == BATCH_SIZE:
                batch_data = cur_batch_data
                batch_label = cur_batch_label
                batch_mask = cur_batch_mask
                batch_labelweights = cur_batch_labelweights
            else:
                batch_data[0:(cur_batch_size)] = cur_batch_data
                batch_label[0:(cur_batch_size)] = cur_batch_label
                batch_mask[0:(cur_batch_size)] = cur_batch_mask
                batch_labelweights[0:(cur_batch_size)] = cur_batch_labelweights

            # ---------------------------------------------------------------------
            # ---- INFERENCE BELOW ----
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['labelweights_pl']: batch_labelweights,
                         ops['masks_pl']: batch_mask,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            # ---- INFERENCE ABOVE ----
            # ---------------------------------------------------------------------

            pred_val = np.argmax(pred_val, 3) # BxFxN
            cur_pred_val = pred_val[0:cur_batch_size]
            if FLAGS.save:
                if not os.path.exists(FLAGS.save_dir):
                    os.makedirs(FLAGS.save_dir, exist_ok=True)
                for b in range(cur_batch_size):
                    for f in range(FLAGS.num_frame):
                        seq_name = fns[b][f][0]
                        frame_id = fns[b][f][1]
                        file_dir = os.path.join(FLAGS.save_dir, seq_name + '-' + str(frame_id).zfill(6) + '.npz')
                        print('Saving: ', file_dir)
                        
                        np.savez(file_dir,
                            point=cur_batch_data[b, f, :, :3],
                            rgb=cur_batch_data[b, f, :, 3:],
                            label=cur_batch_label[b, f], 
                            prediction=cur_batch_label[b, f])
            
            if cur_batch_size == BATCH_SIZE:
                loss_sum += loss_val
            for l in range(NUM_CLASSES):
                total_pred_label_class[l] += np.sum(((cur_pred_val==l) | (cur_batch_label==l)) & cur_batch_mask)
                total_correct_class[l] += np.sum((cur_pred_val==l) & (cur_batch_label==l) & cur_batch_mask)
                total_class[l] += np.sum((cur_batch_label==l) & cur_batch_mask)

    log_test_string('eval mean loss: %f' % (loss_sum / float(len(dataset)/BATCH_SIZE)))

    ACCs = []
    for i in range(NUM_CLASSES):
        if total_class[i] == 0:
            acc = 0
        else:
            acc = total_correct_class[i] / float(total_class[i])
        
        log_test_string('eval acc of %s:\t %f'%(class_mapping.index_to_class[class_mapping.label_to_index[i]], acc))
        ACCs.append(acc)
    
    log_test_string('eval accuracy: %f'% (np.mean(np.array(ACCs))))

    IoUs = []
    for i in range(NUM_CLASSES):
        if total_pred_label_class[i] == 0:
            iou = 0
        else:
            iou = total_correct_class[i] / float(total_pred_label_class[i])
        
        log_test_string('eval mIoU of %s:\t %f'%(class_mapping.index_to_class[class_mapping.label_to_index[i]], iou))
        IoUs.append(iou)
    
    log_test_string('eval mIoU:\t %f'%(np.mean(np.array(IoUs))))

    return loss_sum/float(len(dataset)/BATCH_SIZE)


if __name__ == "__main__":
    log_test_string('pid: %s'%(str(os.getpid())))
    test()
    LOG_TEST_FOUT.close()
