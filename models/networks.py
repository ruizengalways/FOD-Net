import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def define_network(init_type='normal', init_gain=1., gpu_ids=[]):
    """Create the model
    
    Get the FODNet architecture
    """
    net = None
    net = Sep9ResdiualDeeperBN()

    return init_net(net, init_type, init_gain, gpu_ids)


class Sep9ResdiualDeeperBN(nn.Module):
    """Baseline moduel
    """
    def __init__(self):
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



def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        init_weights(net, init_type, init_gain=init_gain, activation='leaky_relu')
    return net


def init_weights(net, init_type='xavier', init_gain=1.0, activation='relu'):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=activation)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
