import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from resnetcopy import resnet34
import numpy as np
from denseASPP import daspp_module
from attention import casa2, scSE


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class aspp_module(nn.Module):
    def __init__(self, in_planes, out_planes, r1=2, r2=4):
        super(aspp_module, self).__init__()
        self.conv = BasicConv2d(in_planes, in_planes, kernel_size=3, padding=1)
        inter_planes = in_planes//4
        self.branch_1 = nn.Sequential(BasicConv2d(in_planes, inter_planes, kernel_size=1),
                                      BasicConv2d(inter_planes, inter_planes, kernel_size=3, padding=1))
        self.branch_2 = nn.Sequential(BasicConv2d(in_planes, inter_planes, kernel_size=1),
                                      BasicConv2d(inter_planes, inter_planes, kernel_size=3, padding=r1, dilation=r1))
        self.branch_3 = nn.Sequential(BasicConv2d(in_planes, inter_planes, kernel_size=1),
                                      BasicConv2d(inter_planes, inter_planes, kernel_size=3, padding=r2, dilation=r2))
        self.branch_4 = BasicConv2d(3*inter_planes, out_planes, kernel_size=1)

    def forward(self,x):
        x = self.conv(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        out = self.branch_4(torch.cat([x1,x2,x3], 1))
        return out


class CMFM(nn.Module):
    def __init__(self, channel):
        super(CMFM, self).__init__()
        self.conv0 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.pp = daspp_module(channel,channel)
        # self.cga = CGA(channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(channel, channel // 4, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(channel // 4, channel, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv2d(channel, channel // 4, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(channel // 4, channel, 1, bias=False))

        self.convc = BasicConv2d(channel*3, channel, 1, padding=0)
        self.convc2 = BasicConv2d(channel*2, channel, 1, padding=0)
        self.conv = BasicConv2d(channel, channel, 1, padding=0)
        self.conv2 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, r, d):
        fd2 = self.conv0(d)
        fr2 = self.conv1(r)
        cat2 = self.convc2(torch.cat([fd2,fr2],1))
        avg_out2 = self.fc1(self.avg_pool(cat2))
        max_out2 = self.fc2(self.max_pool(cat2))
        frd = cat2*torch.sigmoid(max_out2+avg_out2)
        # frd = self.cga(fr, d)
        fd = self.pp(d)
        fr = self.pp(r)
        add = self.conv(fd*fr)
        fd = fd + add
        fr = fr + add
        # al = fd+fr+frd
        # ard = (fd+frd)*fr
        cat = self.convc(torch.cat([frd,fd,fr],1))
        avg_out = torch.mean(cat, dim=1, keepdim=True)

        max_out, _ = torch.max(cat, dim=1, keepdim=True)
        # print(max_out.shape)
        cm = torch.sigmoid(self.conv2(torch.cat([avg_out, max_out], dim=1)))
        cm = cat*cm
        return cm


class FF(nn.Module):
    def __init__(self,  channel):
        super(FF, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), BasicConv2d(inchannel, channel, 3, padding=1))
        self.conv2 = BasicConv2d(channel*2, channel, 3, padding=1)
        # self.conv3 = BasicConv2d(channel, outchannel, 3, padding=1)

    def forward(self, cm):
        ff = self.conv1(cm*torch.sigmoid(1-cm))
        cm = self.conv2(torch.cat([ff, cm], 1))
        # fj01 = self.conv2(fj0)
        # cm = self.conv3(F.interpolate(cm, scale_factor=2, mode='bilinear'))
        # fj = self.conv4(cm*fj01)+fj02
        return cm


class FF2(nn.Module):
    def __init__(self, channel, outchannel):
        super(FF2, self).__init__()
        self.conv1 = BasicConv2d(outchannel, outchannel, 3, padding=1)
        # self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), BasicConv2d(inchannel, channel, 3, padding=1))
        self.conv2 = BasicConv2d(outchannel * 2, outchannel, 3, padding=1)
        self.conv3 = BasicConv2d(channel, outchannel, 3, padding=1)

    def forward(self, cm, cm1):
        cm1 = self.conv3(F.interpolate(cm1, scale_factor=2, mode='bilinear'))
        ff = self.conv1(cm * torch.sigmoid(1 - cm1))
        cm = self.conv2(torch.cat([ff, cm], 1))+cm1
        # fj01 = self.conv2(fj0)
        # fj = self.conv4(cm*fj01)+fj02
        return cm

# class MF(nn.Module):
#     def __init__(self, inchannel, channel):
#         super(MF, self).__init__()
#         self.conv1 = BasicConv2d(channel, channel, 1, padding=0)
#         self.conv2 = nn.Sequential(BasicConv2d(channel, channel, 1, padding=0), scSE(channel))
#         self.conv3 = BasicConv2d(inchannel, channel, 3, padding=1)
#
#     def forward(self, cm, ff, s0):
#         cm = self.conv1(cm)
#         s0 = self.conv3(F.interpolate(s0, cm.size()[2:], mode='bilinear', align_corners=True))
#         s = self.conv2(ff+cm+s0)
#         return s


class Circle(nn.Module):
    def __init__(self, channel):
        super(Circle, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv = nn.Sequential(nn.AdaptiveAvgPool2d(1), BasicConv2d(inchannel, channel, 3, padding=1))

    def forward(self,cm1,cm2,cm3,cm4,cm5):
        cm1 = self.conv1(cm1)
        cm2 = self.conv2(cm2)
        cm3 = self.conv3(cm3)
        cm4 = self.conv4(cm4)
        cm5 = self.conv5(cm5)
        s1 = torch.sigmoid(cm1+cm1*(F.interpolate(cm2, scale_factor=2, mode='bilinear')
                      + F.interpolate(cm3, scale_factor=4, mode='bilinear')
                      + F.interpolate(cm4, scale_factor=8, mode='bilinear')
                      + F.interpolate(cm5, scale_factor=16, mode='bilinear')))
        s2 = torch.sigmoid(cm2+cm2*(F.interpolate(cm1, scale_factor=1/2, mode='bilinear')
                      + F.interpolate(cm3, scale_factor=2, mode='bilinear')
                      + F.interpolate(cm4, scale_factor=4, mode='bilinear')
                      + F.interpolate(cm5, scale_factor=8, mode='bilinear')))
        s3 = torch.sigmoid(cm3+cm3*(F.interpolate(cm1, scale_factor=1/4, mode='bilinear')
                      + F.interpolate(cm2, scale_factor=1/2, mode='bilinear')
                      + F.interpolate(cm4, scale_factor=2, mode='bilinear')
                      + F.interpolate(cm5, scale_factor=4, mode='bilinear')))
        s4 = torch.sigmoid(cm4+cm4*(F.interpolate(cm1, scale_factor=1/8, mode='bilinear')
                      + F.interpolate(cm2, scale_factor=1/4, mode='bilinear')
                      + F.interpolate(cm3, scale_factor=1/2, mode='bilinear')
                      + F.interpolate(cm5, scale_factor=2, mode='bilinear')))
        s5 = torch.sigmoid(cm5+cm5*(F.interpolate(cm1, scale_factor=1/16, mode='bilinear')
                      + F.interpolate(cm2, scale_factor=1/8, mode='bilinear')
                      + F.interpolate(cm3, scale_factor=1/4, mode='bilinear')
                      + F.interpolate(cm4, scale_factor=1/2, mode='bilinear')))
        return s1,s2,s3,s4,s5


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class MMNet(nn.Module):
    def __init__(self, channel=32):
        super(MMNet, self).__init__()
        self.resnet = resnet34()
        self.resnet2 = resnet34()
        self.cmfm1 = CMFM(64)
        self.cmfm2 = CMFM(64)
        self.cmfm3 = CMFM(128)
        self.cmfm4 = CMFM(256)
        self.cmfm5 = CMFM(512)
        self.conv = BasicConv2d(512, 512, 3, padding=1)
        # self.ff5 = FF(512, 256)
        self.ff4 = FF(512)
        self.ff3 = FF2(512, 256)
        self.ff2 = FF2(256, 128)
        self.ff1 = FF2(128, 64)
        self.ff0 = FF2(64, 64)
        # self.mf1 = MF1(64)
        # self.mf2 = MF(64, 64)
        # self.mf3 = MF(64, 128)
        # self.mf4 = MF(128, 256)
        # self.mf5 = MF(256, 512)

        self.reduce_s1 = Reduction(64, channel)
        self.reduce_s2 = Reduction(64, channel)
        self.reduce_s3 = Reduction(128, channel)
        self.reduce_s4 = Reduction(256, channel)
        self.reduce_s5 = Reduction(512, channel)
        self.circle = Circle(32)
        self.decode = nn.Conv2d(5,1,3,1,1)
        # self.decode1 = nn.Conv2d(64,1,3,1,1)
        # self.decode2 = nn.Conv2d(128, 1, 3, 1, 1)
        # self.decode3 = nn.Conv2d(256, 1, 3, 1, 1)
        # self.decode4 = nn.Conv2d(512, 1, 3, 1, 1)
        # self.decode5 = nn.Conv2d(512, 1, 3, 1, 1)
        self.decode1 = nn.Conv2d(32,1,3,1,1)
        self.decode2 = nn.Conv2d(32, 1, 3, 1, 1)
        self.decode3 = nn.Conv2d(32, 1, 3, 1, 1)
        self.decode4 = nn.Conv2d(32, 1, 3, 1, 1)
        self.decode5 = nn.Conv2d(32, 1, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.initialize_weights()

    def forward(self, rgb, depth):
        rgb_size = rgb.shape[2:]
        rgb1, rgb2, rgb3, rgb4, rgb5 = self.resnet(rgb)
        depth1, depth2, depth3, depth4, depth5 = self.resnet2(depth)
        cm1 = self.cmfm1(rgb1, depth1)
        cm2 = self.cmfm2(rgb2, depth2)
        cm3 = self.cmfm3(rgb3, depth3)
        cm4 = self.cmfm4(rgb4, depth4)
        cm5 = self.cmfm5(rgb5, depth5)
        ff5 = self.conv(cm5)
        ff4 = self.ff4(ff5)
        ff3 = self.ff3(cm4, ff4)
        ff2 = self.ff2(cm3, ff3)
        ff1 = self.ff1(cm2, ff2)
        ff0 = self.ff0(cm1, ff1)
        # s1 = self.mf1(cm1, ff1)
        # s2 = self.mf2(cm2, ff2, s1)
        # s3 = self.mf3(cm3, ff3, s2)
        # s4 = self.mf4(cm4, ff4, s3)
        # s5 = self.mf5(cm5, ff5, s4)
        x_1 = self.reduce_s1(ff0)
        x_2 = self.reduce_s2(ff1)
        x_3 = self.reduce_s3(ff2)
        x_4 = self.reduce_s4(ff3)
        x_5 = self.reduce_s5(ff4)
        s1,s2,s3,s4,s5, = self.circle(x_1,x_2,x_3,x_4,x_5)
        features_5 = self.decode5(s5)
        features_4 = self.decode4(s4)
        features_3 = self.decode3(s3)
        features_2 = self.decode2(s2)
        features_1 = self.decode1(s1)

        features_5 = F.interpolate(features_5, rgb_size, mode='bilinear', align_corners=True)
        features_4 = F.interpolate(features_4, rgb_size, mode='bilinear', align_corners=True)
        features_3 = F.interpolate(features_3, rgb_size, mode='bilinear', align_corners=True)
        features_2 = F.interpolate(features_2, rgb_size, mode='bilinear', align_corners=True)
        features_1 = F.interpolate(features_1, rgb_size, mode='bilinear', align_corners=True)
        pred = torch.cat((features_2,features_3,features_4,features_5,features_1), dim=1)
        pred = self.decode(pred)
        if self.training:
            return pred, features_1, features_2, features_3, features_4, features_5
        return pred
        # return features_5

    def initialize_weights(self):
        res34 = models.resnet34(pretrained=True)
        self.resnet.load_state_dict(res34.state_dict(), False)
        self.resnet2.load_state_dict(res34.state_dict(), False)

if __name__ =="__main__":
    rgb = torch.randn((2, 3, 224, 224))
    depth = torch.randn(2, 3, 224, 224)
    net = MMNet()
    outputs = net(rgb,depth)
    # print(outputs.shape)
    print([i.size() for i in outputs])