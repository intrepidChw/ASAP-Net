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
from tools.sync_batchnorm.batchnorm import convert_model
import argparse
import importlib
import yaml
import time

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--ckpt_save_interval", type=int, default=1)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument("--ckpt", type=str, default='None')

parser.add_argument("--net", type=str, default='PNv2_ASAP-2')
parser.add_argument('--frame_num', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.012)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[5, 10, 15, 20, 25])
parser.add_argument('--weight_decay', type=float, default=0.0001)

parser.add_argument("--data_root", type=str, default='/mnt/sdd/hanwen/dataset/semanticKITTI/dataset')
parser.add_argument("--data_config", type=str, default='config/labels/semantic-kitti.yaml')
parser.add_argument("--output_dir", type=str, default='/mnt/sdd/hanwen/SemKITTI_experiments/PointNet2/test')
parser.add_argument("--extra_tag", type=str, default='untested')

args = parser.parse_args()

FG_THRESH = 0.3


def log_print(info, log_f=None):
    print(info)
    if log_f is not None:
        print(info, file=log_f)


# class DiceLoss(nn.Module):
#     def __init__(self, ignore_target=-1):
#         super().__init__()
#         self.ignore_target = ignore_target
#
#     def forward(self, input, target):
#         """
#         :param input: (N), logit
#         :param target: (N), {0, 1}
#         :return:
#         """
#         input = torch.sigmoid(input.view(-1))
#         target = target.float().view(-1)
#         mask = (target != self.ignore_target).float()
#         return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(), min=1.0)

class DiceLoss(nn.Module):
    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       mask = (tflat != self.ignore_target).float()
       intersection = (iflat * tflat * mask).sum()
       A_sum = torch.sum(iflat * iflat * mask)
       B_sum = torch.sum(tflat * tflat * mask)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def train_one_epoch(model, train_loader, optimizer, epoch, lr_scheduler, total_it, tb_log, log_f, loss_func):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()
    update_ratio_meter = AverageMeter()

    model.train()
    log_print('===============TRAIN EPOCH %d================' % epoch, log_f=log_f)
    # loss_func = DiceLoss(ignore_target=-1)
    # print('len:', len(train_loader))
    torch.cuda.empty_cache()
    end = time.time()
    for it, (_, _, _, _, _, _, _, xyz_seq, range_seq, remission_seq, labels_seq) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        # pts_input, cls_labels = batch['pts_input'], batch['cls_labels']
        # pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        # cls_labels = torch.from_numpy(cls_labels).cuda(non_blocking=True).long().view(-1)
        xyz_seq = xyz_seq.cuda(non_blocking=True).float()
        labels_seq = labels_seq.cuda(non_blocking=True).long()
        remission_seq = remission_seq.cuda(non_blocking=True).float()
        range_seq = range_seq.cuda(non_blocking=True).float()
        # print('xyz_seq:', xyz_seq.shape)
        features = torch.cat([remission_seq.unsqueeze(-1), range_seq.unsqueeze(-1)], dim=-1).transpose(2, 3)
        pred_cls = model(xyz_seq, features)
        # xentropy
        pred_cls = pred_cls.transpose(1, 2)
        # print('pred_cls:', pred_cls.shape)
        # print('labels_seq:', labels_seq.shape)
        loss = loss_func(torch.log(pred_cls.clamp(min=1e-8)), labels_seq)

        # # dice
        # pred_cls = pred_cls.clamp(min=1e-8)
        # pred_cls = pred_cls.transpose(1, 2)
        # labels_seq_loss = labels_seq.unsqueeze(1)
        # labels_seq_loss = torch.zeros(pred_cls.shape).cuda().scatter_(1, labels_seq_loss, 1).float()
        # # print('pred_cls:', pred_cls.shape)
        # # print('labels_seq_loss:', labels_seq_loss.shape)
        # loss = loss_func(pred_cls, labels_seq_loss)

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_it += 1

        # measure accuracy and record loss
        loss = loss.mean()
        with torch.no_grad():
            # evaluator.reset()
            argmax = pred_cls.argmax(dim=1)
            # print('argmax:', argmax.shape)
            # print('labels_seq:', labels_seq.shape)
            evaluator.addBatch(argmax, labels_seq)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()

        losses.update(loss.item(), xyz_seq.size(0))
        acc.update(accuracy.item(), xyz_seq.size(0))
        iou.update(jaccard.item(), xyz_seq.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # get gradient updates and weights, so I can print the relationship of
        # their norms
        update_ratios = []
        for g in optimizer.param_groups:
            lr = g["lr"]
            for value in g["params"]:
                if value.grad is not None:
                    w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                    update = np.linalg.norm(-max(lr, 1e-10) *
                                            value.grad.cpu().numpy().reshape((-1)))
                    update_ratios.append(update / max(w, 1e-10))
        update_ratios = np.array(update_ratios)
        update_mean = update_ratios.mean()
        update_std = update_ratios.std()
        update_ratio_meter.update(update_mean)  # over the epoch

        cur_lr = lr_scheduler.get_lr()[0]
        # tb_log.log_value('learning_rate', cur_lr, epoch)
        # if tb_log is not None:
        #     tb_log.log_value('train_loss', loss, total_it)
        #     tb_log.log_value('train_fg_iou', iou, total_it)
        #
        # log_print('training epoch %d: it=%d/%d, total_it=%d, loss=%.5f, fg_iou=%.3f, lr=%f' %
        #           (epoch, it, len(train_loader), total_it, loss.item(), iou.item(), cur_lr), log_f=log_f)
        print('Lr: {lr:.3e} | '
              'Update: {umean:.3e} mean,{ustd:.3e} std | '
              'Epoch: [{0}][{1}/{2}] | '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
              'acc {acc.val:.3f} ({acc.avg:.3f}) | '
              'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                epoch, it, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc, iou=iou, lr=cur_lr,
                umean=update_mean, ustd=update_std))

    jaccard, class_jaccard = evaluator.getIoU()

    return acc.avg, iou.avg, class_jaccard, losses.avg, update_ratio_meter.avg


def eval_one_epoch(model, eval_loader, epoch, tb_log, log_f, loss_func, class_func):
    model.eval()
    log_print('===============EVAL EPOCH %d================' % epoch, log_f=log_f)
    # loss_func = DiceLoss(ignore_target=-1)
    # loss_func = nn.NLLLoss(loss_w).cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for it, (_, _, _, _, _, _, _, xyz_seq, range_seq, remission_seq, labels_seq) in enumerate(eval_loader):
            # pts_input, cls_labels = batch['pts_input'], batch['cls_labels']
            # pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
            # cls_labels = torch.from_numpy(cls_labels).cuda(non_blocking=True).long().view(-1)
            xyz_seq = xyz_seq.cuda(non_blocking=True).float()
            labels_seq = labels_seq.cuda(non_blocking=True).long()
            remission_seq = remission_seq.cuda(non_blocking=True).float()
            range_seq = range_seq.cuda(non_blocking=True).float()

            # features = torch.cat([xyz_seq, remission_seq.unsqueeze(-1), range_seq.unsqueeze(-1)], dim=-1).transpose(2, 3)
            features = torch.cat([remission_seq.unsqueeze(-1), range_seq.unsqueeze(-1)], dim=-1).transpose(2, 3)

            pred_cls = model(xyz_seq, features)
            # cross entropy
            pred_cls = pred_cls.transpose(1, 2)
            loss = loss_func(torch.log(pred_cls.clamp(min=1e-8)), labels_seq)

            # # dice
            # pred_cls = pred_cls.clamp(min=1e-8)
            # pred_cls = pred_cls.transpose(1, 2)
            # labels_seq_loss = labels_seq.unsqueeze(1)
            # labels_seq_loss = torch.zeros(pred_cls.shape).cuda().scatter_(1, labels_seq_loss, 1).float()
            # # print('pred_cls:', pred_cls.shape)
            # # print('labels_seq_loss:', labels_seq_loss.shape)
            # loss = loss_func(pred_cls, labels_seq_loss)

            # measure accuracy and record loss
            loss = loss.mean()
            # evaluator.reset()
            argmax = pred_cls.argmax(dim=1)
            evaluator.addBatch(argmax, labels_seq)
            losses.update(loss.item(), xyz_seq.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), xyz_seq.size(0))
        iou.update(jaccard.item(), xyz_seq.size(0))
        print('Validation finished.')

        print('Validation set:\n'
              'Time avg per batch {batch_time.avg:.3f}\n'
              'Loss avg {loss.avg:.4f}\n'
              'Acc avg {acc.avg:.3f}\n'
              'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time,
                                             loss=losses,
                                             acc=acc, iou=iou))
        # print also classwise
        for i, jacc in enumerate(class_jaccard):
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_func(i), jacc=jacc))

    return acc.avg, iou.avg, class_jaccard, losses.avg


def save_checkpoint(model, epoch, ckpt_name):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    state = {'epoch': epoch, 'model_state': model_state}
    ckpt_name = '{}.pth'.format(ckpt_name)
    torch.save(state, ckpt_name)


def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        log_print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        w_dict = checkpoint['model_state']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        w_keys = w_dict.keys()
        for key in model_keys:
            if key in w_keys:
                model_dict[key] = w_dict[key]
        model.load_state_dict(model_dict, strict=True)
        # model.load_state_dict(checkpoint['model_state'])
        log_print("==> Done")
    else:
        raise FileNotFoundError

    return epoch


def train_and_eval(model, train_set, eval_loader, tb_log, ckpt_dir, log_f, loss_func, class_func):
    best_train_iou = 0.0
    best_val_iou = 0.0

    model = convert_model(model).cuda()
    # model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in args.decay_step_list:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * args.lr_decay
        return max(cur_decay, args.lr_clip / args.lr)

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)

    total_it = 0
    for epoch in range(1, args.epochs + 1):
        # train
        train_set.data_reset()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=args.workers)
        
        lr_scheduler.step(epoch)
        acc, miou_train, ious, loss, update_mean = train_one_epoch(model, train_loader, optimizer, epoch, lr_scheduler, total_it,
                                                      tb_log, log_f, loss_func)

        cur_lr = lr_scheduler.get_lr()[0]
        tb_log.add_scalar('lr', cur_lr, epoch)
        tb_log.add_scalar('train_acc', acc, epoch)
        tb_log.add_scalar('train_loss', loss, epoch)
        tb_log.add_scalar('train_iou/mIoU', miou_train, epoch)
        for i, iou in enumerate(ious):
            # print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            #     i=i, class_str=class_func(i), jacc=jacc))
            tb_log.add_scalar('train_iou/' + class_func(i), iou, epoch)
        
        if miou_train > best_train_iou:
            print("Best mean iou in train so far, save model!")
            print("*" * 80)
            best_train_iou = miou_train
            # ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % epoch
            ckpt_name = os.path.join(ckpt_dir, 'checkpoint_train')
            save_checkpoint(model, epoch, ckpt_name)

        # evaluation
        acc, miou_eval, ious, loss = eval_one_epoch(model, eval_loader, epoch, tb_log, log_f, loss_func, class_func)

        tb_log.add_scalar('eval_acc', acc, epoch)
        # tb_log.add_scalar('eval_iou', iou, epoch)
        tb_log.add_scalar('eval_loss', loss, epoch)
        tb_log.add_scalar('eval_iou/mIoU', miou_eval, epoch)
        for i, iou in enumerate(ious):
            # print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            #     i=i, class_str=class_func(i), jacc=jacc))
            tb_log.add_scalar('eval_iou/' + class_func(i), iou, epoch)

        # remember best iou and save checkpoint
        if miou_eval > best_val_iou:
            print("Best mean iou in validation so far, save model!")
            print("*" * 80)
            best_val_iou = miou_eval
            # ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % epoch
            ckpt_name = os.path.join(ckpt_dir, 'checkpoint_eval')
            save_checkpoint(model, epoch, ckpt_name)


DATA_ROOT = args.data_root
DATA_CONFIG = args.data_config
# ARCH_CONFIG = 'config/arch/squeezesegV2TU_crf.yaml'

if __name__ == '__main__':
    MODEL = importlib.import_module(args.net)  # import network module
    model = MODEL.get_model(input_channels=2)
    model = convert_model(model)
    if args.ckpt != 'None':
        epoch = load_checkpoint(model, args.ckpt)
        print('Resume from epoch %d...' % epoch)
    model = nn.DataParallel(model)   # spread in gpus
    device = "cuda"
    # # open arch config file
    # try:
    #     print("Opening arch config file %s" % ARCH_CONFIG)
    #     ARCH = yaml.safe_load(open(ARCH_CONFIG, 'r'))
    # except Exception as e:
    #     print(e)
    #     print("Error opening arch yaml file.")
    #     quit()

    # open data config file
    try:
        print("Opening data config file %s" % DATA_CONFIG)
        DATA = yaml.safe_load(open(DATA_CONFIG, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    nclasses = len(DATA["learning_map_inv"])
    epsilon_w = 0.001
    content = torch.zeros(nclasses, dtype=torch.float)
    for cl, freq in DATA["content"].items():
        x_cl = SemanticKitti.map(cl, DATA["learning_map"])  # map actual class to xentropy class
        content[x_cl] += freq
    loss_w = 1 / (content + epsilon_w)  # get weights
    for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
        if DATA["learning_ignore"][x_cl]:
            # don't weigh
            loss_w[x_cl] = 0
    print("Loss weights from content: ", loss_w.data)

    loss_func = nn.NLLLoss(weight=loss_w).to(device)
    # loss_func = DiceLoss(ignore_target=-1)

    eval_set = SemanticKitti(root=DATA_ROOT,
                             sequences=DATA["split"]["valid"],
                             labels=DATA["labels"],
                             color_map=DATA["color_map"],
                             learning_map=DATA["learning_map"],
                             learning_map_inv=DATA["learning_map_inv"],
                             sensor=DATA["dataset"]["sensor"],
                             mode='eval',
                             frame_num=args.frame_num,
                             sample_points=45000,
                             gt=True)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, pin_memory=True,
                             num_workers=args.workers)

    train_set = SemanticKitti(root=DATA_ROOT,
                              sequences=DATA["split"]["train"],
                              labels=DATA["labels"],
                              color_map=DATA["color_map"],
                              learning_map=DATA["learning_map"],
                              learning_map_inv=DATA["learning_map_inv"],
                              sensor=DATA["dataset"]["sensor"],
                              mode='train',
                              frame_num=args.frame_num,
                              sample_points=45000,
                              gt=True)

    # train_set = KittiDataset(root_dir='./data', mode='TRAIN', split='train')
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
    #                           num_workers=args.workers)

    ignore_class = [0]
    evaluator = iouEval(nclasses, device, ignore_class)

    # output dir config
    output_dir = os.path.join(args.output_dir, args.extra_tag)
    os.makedirs(output_dir, exist_ok=True)
    tb_log = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'log.txt')
    log_f = open(log_file, 'w')

    for key, val in vars(args).items():
        log_print("{:16} {}".format(key, val), log_f=log_f)

    # train and eval
    train_and_eval(model, train_set, eval_loader, tb_log, ckpt_dir, log_f, loss_func,
                   class_func=eval_set.get_xentropy_class_string)
    log_f.close()

