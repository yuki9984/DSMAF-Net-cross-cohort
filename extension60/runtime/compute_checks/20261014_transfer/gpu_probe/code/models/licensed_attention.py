# SPDX-License-Identifier: MIT
# Attention adapted from CASIA-IVA-Lab/DANet, b0a0fa008148f102321cdaa7ca38fb0ddd0239d1.
# Created by CASIA IVA, copyright (c) 2018. See LICENSES/DANet-MIT.txt.
# Changes: full-width depthwise Q/K/V and a documented historical-compatible head.
import torch
from torch import nn
from torch.nn import Module, Parameter, Softmax


class DepthWiseConv2d(nn.Module):
    """Depthwise then pointwise projection; historical parameter names retained."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                              dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.conv1(x))

class PAM_Module(Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = DepthWiseConv2d(in_dim, in_dim)
        self.key_conv = DepthWiseConv2d(in_dim, in_dim)
        self.value_conv = DepthWiseConv2d(in_dim, in_dim)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class DA_Block(nn.Module):
    """Checkpoint-compatible dual-branch head; not a new research method."""
    def __init__(self, in_channels):
        super().__init__()
        c = in_channels // 16
        def projection(a,b,k):
            return nn.Sequential(DepthWiseConv2d(a,b,k,padding=k//2),nn.ReLU())
        self.conv5a = projection(in_channels,c,3)
        self.conv5c = projection(in_channels,c,3)
        self.sa = PAM_Module(c)
        self.sc = CAM_Module(c)
        self.conv51 = projection(c,c,3)
        self.conv52 = projection(c,c,3)
        def output(a,b):
            return nn.Sequential(nn.Dropout2d(0.05,False),DepthWiseConv2d(a,b),nn.ReLU())
        self.conv6 = output(c,in_channels)
        self.conv7 = output(c,in_channels)
        self.conv8 = output(in_channels,in_channels)
    def forward(self,x):
        position = self.conv6(self.conv51(self.sa(self.conv5a(x))))
        channel = self.conv7(self.conv52(self.sc(self.conv5c(x))))
        return self.conv8(position+channel)
