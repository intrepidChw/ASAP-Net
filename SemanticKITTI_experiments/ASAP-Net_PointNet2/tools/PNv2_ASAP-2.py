import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
import pointnet2.pytorch_utils as pt_utils
import pointnet2.pointnet2_utils as pointnet2_utils
import torch.nn.functional as F


def get_model(input_channels=0, do_interpolation=False):
    return Pointnet2MSG(input_channels=input_channels, do_interpolation=do_interpolation)


NPOINTS = [16384, 4096, 1024, 256]
# RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
# RADIUS = [[0.5, 1.0], [1.25, 2.5], [8.0, 16.0], [16.0, 32.0]]
RADIUS = [[0.5, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
CLS_FC = [128]
DP_RATIO = 0.5

MLP1 = [256, 256, 512+256]
FP_MLP1 = [512 + 512 + 256, 256]
RADIUS1 = 3.0
NPOINT1 = 1024
NSAMPLE1 = 32

MLP2 = [512, 512, 1024+512]
FP_MLP2 = [1024 + 512, 512]
RADIUS2 = 6.0
NPOINT2 = 256
NSAMPLE2 = 32


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6, do_interpolation=False):
        super().__init__()
        print('PNv2_ASAP-2')
        self.do_interpolation = do_interpolation
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.tem_sa1 = PointnetSAModule(mlp=MLP1, npoint=NPOINT1, nsample=NSAMPLE1, radius=RADIUS1)
        self.fuse_layer1_atten = nn.Conv1d(MLP1[-1]*2, 2, kernel_size=1, stride=1)
        self.fuse_layer1 = nn.Conv1d(MLP1[-1], 512, kernel_size=1, stride=1)
        self.tem_fp1 = PointnetFPModule(mlp=FP_MLP1)

        self.tem_sa2 = PointnetSAModule(mlp=MLP2, npoint=NPOINT2, nsample=NSAMPLE2, radius=RADIUS2)
        self.fuse_layer2_atten = nn.Conv1d(MLP2[-1]*2, 2, kernel_size=1, stride=1)
        self.fuse_layer2 = nn.Conv1d(MLP2[-1], 1024, kernel_size=1, stride=1)
        self.tem_fp2 = PointnetFPModule(mlp=FP_MLP2)

        self.tem_dropout = nn.Dropout(DP_RATIO)

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k])
            )

        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 20, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, xyz_seq: torch.cuda.FloatTensor, features_seq: torch.cuda.FloatTensor, full_xyz_seq=None):
        out_list = []
        frame_num = xyz_seq.shape[1]
        for frame_idx in range(frame_num):
            
            xyz, features = xyz_seq[:, frame_idx, ...].contiguous(), features_seq[:, frame_idx, ...].contiguous()
            if self.do_interpolation:
                full_xyz = full_xyz_seq[:, frame_idx, ...].contiguous()

            l_xyz, l_features = [xyz], [features]
            for i in range(2):
                li_xyz, li_features, _ = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)

            cur_xyz_l0 = l_xyz[-1]
            cur_feat_l0 = l_features[-1]

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
            # cur_x = self.tem_fp1(cur_xyz_l0, cur_xyz_l1, cur_feat_l0, feat_fp1)

            l_features[-1] = self.tem_fp1(cur_xyz_l0, cur_xyz_l1, cur_feat_l0, feat_fp1)

            for i in range(2, len(self.SA_modules)):
                li_xyz, li_features, _ = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)

            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )

            pred_cls = self.cls_layer(l_features[0]).contiguous()  # (B, 20, N)
            if self.do_interpolation:
                pred_cls = self.interpolate(xyz, full_xyz, pred_cls)
            
            out_list.append(pred_cls.unsqueeze(1))
            # print('pred_cls:', pred_cls.shape)
        out = torch.cat(out_list, dim=1)
        out = F.softmax(out, dim=2)

        return out

    def interpolate(self, xyz, full_xyz, known_feats):
        dist, idx = pointnet2_utils.three_nn(full_xyz, xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        return interpolated_feats
