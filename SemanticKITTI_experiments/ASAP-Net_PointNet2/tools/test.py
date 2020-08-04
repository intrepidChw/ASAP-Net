import _init_path
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from dataset import KittiDataset
from tools.semanticKITTI import SemanticKitti
from tools.common.avgmeter import AverageMeter
from tools.ioueval import iouEval
import argparse
import importlib
import yaml
import time
from thop import profile


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt_save_interval", type=int, default=1)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument("--ckpt", type=str, default='None')

parser.add_argument("--net", type=str, default='PNv2_ASAP-1')
parser.add_argument('--frame_num', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.008)
parser.add_argument('--lr_decay', type=float, default=0.2)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[50, 70, 80, 90])
parser.add_argument('--weight_decay', type=float, default=0.001)

parser.add_argument("--data_root", type=str, default='/mnt/sdd/hanwen/dataset/semanticKITTI/dataset')
parser.add_argument("--data_config", type=str, default='config/labels/semantic-kitti.yaml')
parser.add_argument("--output_dir", type=str, default='/mnt/sdd/hanwen/dataset/semanticKITTI/results/PointNet2')
parser.add_argument("--extra_tag", type=str, default='tpc1_ssg')

args = parser.parse_args()


def log_print(info, log_f=None):
    print(info)
    if log_f is not None:
        print(info, file=log_f)


def test_model(model, test_loader, epoch, logdir, to_orig_fn):
    model.cuda()
    model.eval()
    print('===============TEST EPOCH %d================' % epoch)
    # loss_func = DiceLoss(ignore_target=-1)
    # loss_func = nn.NLLLoss(loss_w).cuda()
    batch_time = AverageMeter()

    total_time = 0
    cnt = 0
    with torch.no_grad():
        end = time.time()
        for it, (_, path_seqs, path_names, _, full_xyz_seq, _, npoints, xyz_seq, range_seq, remission_seq, labels_seq) in enumerate(test_loader):

            full_xyz_seq = full_xyz_seq.cuda(non_blocking=True).float()

            xyz_seq = xyz_seq.cuda(non_blocking=True).float()
            remission_seq = remission_seq.cuda(non_blocking=True).float()
            range_seq = range_seq.cuda(non_blocking=True).float()
            features = torch.cat([remission_seq.unsqueeze(-1), range_seq.unsqueeze(-1)], dim=-1).transpose(2, 3)

            model_start = time.time()
            pred_cls = model(xyz_seq, features, full_xyz_seq)
            
            model_time = time.time() - model_start
            
            total_time += model_time
            cnt += 1
            pred_cls = torch.argmax(pred_cls, dim=-2)
            for batch_idx in range(xyz_seq.shape[0]):
                for frame_idx in range(xyz_seq.shape[1]):
                    point_num = npoints[frame_idx][batch_idx]
                    cur_pred = pred_cls[batch_idx, frame_idx, :point_num]
                    
                    pred_np = cur_pred.cpu().numpy()
                    pred_np = pred_np.reshape((-1)).astype(np.int32)
                    pred_np = to_orig_fn(pred_np)
    
                    path = os.path.join(logdir, "sequences",
                                        path_seqs[frame_idx][batch_idx], "predictions")
                    os.makedirs(path, exist_ok=True)
                    file = os.path.join(path, path_names[frame_idx][batch_idx])
                    pred_np.tofile(file)
    
                    print('Finish testing seq' + path_seqs[frame_idx][batch_idx] +
                          ' frame' + path_names[frame_idx][batch_idx] + ' in ' + str(time.time() - end) + 's')
    
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
    
            print('Time avg per batch {batch_time.avg:.3f}\n'
                  .format(batch_time=batch_time))
    print('Average model time: ', total_time / cnt)


def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        log_print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        log_print("==> Done")
    else:
        raise FileNotFoundError

    return epoch


DATA_ROOT = args.data_root
DATA_CONFIG = args.data_config


if __name__ == "__main__":
    MODEL = importlib.import_module(args.net)  # import network module
    model = MODEL.get_model(input_channels=2, do_interpolation=True)
    if args.ckpt != 'None':
        epoch = load_checkpoint(model, args.ckpt)
        print('Test model from epoch %d...' % epoch)
    
    # open data config file
    try:
        print("Opening data config file %s" % DATA_CONFIG)
        DATA = yaml.safe_load(open(DATA_CONFIG, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()


    test_set = SemanticKitti(root=DATA_ROOT,
                             sequences=DATA["split"]["test"],
                             labels=DATA["labels"],
                             color_map=DATA["color_map"],
                             learning_map=DATA["learning_map"],
                             learning_map_inv=DATA["learning_map_inv"],
                             sensor=DATA["dataset"]["sensor"],
                             mode='test',
                             frame_num=args.frame_num,
                             sample_points=45000,
                             gt=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True,
                             num_workers=args.workers)

    output_dir = os.path.join(args.output_dir, args.extra_tag)
    os.makedirs(output_dir, exist_ok=True)

    def to_original(label):
        return SemanticKitti.map(label, DATA["learning_map_inv"])

    test_model(model, test_loader, epoch, output_dir, to_original)

