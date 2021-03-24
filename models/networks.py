"""
Fiber orientation super resolution
Licensed under the CC BY-NC-SA 4.0 License (see LICENSE for details)
Written by Rui Zeng @ USyd Brain and Mind Centre (r.zeng@outlook.com / rui.zeng@sydney.edu.au)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
def define_network(device=torch.device('cpu')):
    """Create the model
    """
    net = Sep9ResdiualDeeperBN()
    net.to(device)
    return net

##############################################################################
# Network Architecture Classes
##############################################################################

class Sep9ResdiualDeeperBN(nn.Module):
    """Baseline moduel
    """
    def __init__(self, norm_layer=nn.BatchNorm2d, iter_pose=False, solve_method='svd', pose_max_iter=10):
        """Construct a Resnet-based generator
        """
        super(Sep9ResdiualDeeperBN, self).__init__()
        self.conv3d1 = torch.nn.Conv3d(in_channels=45, out_channels=256, kernel_size=3)
        self.bn3d1 = torch.nn.BatchNorm3d(256)
        self.glu3d1 = torch.nn.GLU(dim=1)

        self.conv3d2 = torch.nn.Conv3d(in_channels=128, out_channels=512, kernel_size=3)
        self.bn3d2 = torch.nn.BatchNorm3d(512)
        self.glu3d2 = torch.nn.GLU(dim=1)

        self.conv3d3 = torch.nn.Conv3d(in_channels=256, out_channels=1024, kernel_size=3)
        self.bn3d3 = torch.nn.BatchNorm3d(1024)
        self.glu3d3 = torch.nn.GLU(dim=1)

        self.conv3d4 = torch.nn.Conv3d(in_channels=512, out_channels=2048, kernel_size=3)
        self.bn3d4 = torch.nn.BatchNorm3d(2048)
        self.glu3d4 = torch.nn.GLU(dim=1)

        self.joint_linear = torch.nn.Linear(in_features=1024, out_features=2048)
        self.joint_bn = torch.nn.BatchNorm1d(2048)
        self.joint_glu = torch.nn.GLU(dim=1)

        self.l0_pred = ceblock(num_coeff=2)
        self.l2_pred = ceblock(num_coeff=10)
        self.l4_pred = ceblock(num_coeff=18)
        self.l6_pred = ceblock(num_coeff=26)
        self.l8_pred = ceblock(num_coeff=34)

    def forward(self, fodlr):
        x = self.conv3d1(fodlr)
        x = self.bn3d1(x)
        x = self.glu3d1(x)

        x = self.conv3d2(x)
        x = self.bn3d2(x)
        x = self.glu3d2(x)

        x = self.conv3d3(x)
        x = self.bn3d3(x)
        x = self.glu3d3(x)

        x = self.conv3d4(x)
        x = self.bn3d4(x)
        x = self.glu3d4(x)

        x = x.squeeze()
        x = self.joint_linear(x)
        x = self.joint_bn(x)
        joint = self.joint_glu(x)

        x = self.l0_pred(joint)
        l0_residual = x[:, :1]
        l0_scale = F.sigmoid(x[:, 1:])

        x = self.l2_pred(joint)
        l2_residual = x[:, :5]
        l2_scale = F.sigmoid(x[:, 5:])

        x = self.l4_pred(joint)
        l4_residual = x[:, :9]
        l4_scale = F.sigmoid(x[:, 9:])

        x = self.l6_pred(joint)
        l6_residual = x[:, :13]
        l6_scale = F.sigmoid(x[:, 13:])

        x = self.l8_pred(joint)
        l8_residual = x[:, :17]
        l8_scale = F.sigmoid(x[:, 17:])

        residual = torch.cat([l0_residual, l2_residual, l4_residual, l6_residual, l8_residual], dim=1)
        scale = torch.cat([l0_scale, l2_scale, l4_scale, l6_scale, l8_scale], dim=1)

        fodpred = residual * scale + fodlr[:, :, 4, 4, 4]

        return fodpred


class ceblock(nn.Module):
    def __init__(self, num_coeff):
        super(ceblock, self).__init__()
        self.l_0 = torch.nn.Linear(in_features=1024, out_features=1024)
        self.bn_0 = torch.nn.BatchNorm1d(1024)
        self.glu_0 = torch.nn.GLU(dim=1)
        self.l_1 = torch.nn.Linear(in_features=512, out_features=512)
        self.bn_1 = torch.nn.BatchNorm1d(512)
        self.glu_1 = torch.nn.GLU(dim=1)
        self.l_2 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_2 = torch.nn.BatchNorm1d(512)
        self.glu_2 = torch.nn.GLU(dim=1)
        self.l_3 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_3 = torch.nn.BatchNorm1d(512)
        self.glu_3 = torch.nn.GLU(dim=1)
        self.l_4 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_4 = torch.nn.BatchNorm1d(512)
        self.glu_4 = torch.nn.GLU(dim=1)
        self.pred = torch.nn.Linear(in_features=256, out_features=num_coeff)

    def forward(self, x):
        x = self.l_0(x)
        x = self.bn_0(x)
        x = self.glu_0(x)
        x = self.l_1(x)
        x = self.bn_1(x)
        x = self.glu_1(x)
        x = self.l_2(x)
        x = self.bn_2(x)
        x = self.glu_2(x)
        x = self.l_3(x)
        x = self.bn_3(x)
        x = self.glu_3(x)
        x = self.l_4(x)
        x = self.bn_4(x)
        x = self.glu_4(x)
        x = self.pred(x)
        return x