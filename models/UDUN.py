#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        else:
            m.initialize()



class TSA(nn.Module):
    def __init__(self):
        super(TSA, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(64, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat_trunk, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_trunk)), inplace=True)
        x = self.act(self.conv1(feat_trunk))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x + y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)


class MSA_256(nn.Module):
    def __init__(self):
        super(MSA_256, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat_trunk, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_trunk)), inplace=True)
        x = self.act(self.conv1(feat_trunk))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x + y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)

class MSA_512(nn.Module):
    def __init__(self):
        super(MSA_512, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat_mask, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_mask )), inplace=True)
        x = self.act(self.conv1(feat_mask))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x+y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)



class MSA_1024(nn.Module):
    def __init__(self):
        super(MSA_1024, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

    def forward(self, feat_mask, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_mask)), inplace=True)
        x = self.act(self.conv1(feat_mask))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x+y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)

class UnionDe(nn.Module):
    def __init__(self):
        super(UnionDe, self).__init__()
        self.TSA_0 = TSA()
        self.TSA_1 = TSA()
        self.TSA_2 = TSA()
        self.MSA_3 = MSA_256()
        self.MSA_4 = MSA_512()
        self.MSA_5 = MSA_1024()

        self.conv_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_4_reduce = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16), nn.ReLU(inplace=True))

    def forward(self, feat_trunk, feat_struct):
        mask = self.TSA_0(feat_trunk[0], feat_struct[0])

        temp = self.TSA_1(feat_trunk[1], feat_struct[1])
        maskup = F.interpolate(mask, size=temp.size()[2:], mode='bilinear')
        temp = maskup + temp
        mask = self.conv_1(temp)

        temp = self.TSA_2(feat_trunk[2], feat_struct[2])
        maskup = F.interpolate(mask, size=temp.size()[2:], mode='bilinear')
        temp = maskup + temp
        mask = self.conv_2(temp)

        maskup = F.interpolate(mask, size=feat_struct[3].size()[2:], mode='bilinear')
        temp = self.MSA_3(maskup, feat_struct[3])
        temp = maskup + temp
        mask = self.conv_3(temp)

        maskup = F.interpolate(mask, size=feat_struct[4].size()[2:], mode='bilinear')
        temp = self.MSA_4(maskup, feat_struct[4])
        temp = maskup + temp
        mask = self.conv_4(temp)

        maskup = F.interpolate(mask, size=feat_struct[5].size()[2:], mode='bilinear')
        temp = self.MSA_5(maskup, feat_struct[5])
        maskup = self.conv_4_reduce(maskup)
        temp = maskup + temp
        mask = self.conv_5(temp)

        return mask

    def initialize(self):
        weight_init(self)


class TrunkDe(nn.Module):
    def __init__(self):
        super(TrunkDe, self).__init__()

        # T1
        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_1_2 = nn.BatchNorm2d(64)
        # T2
        self.conv_2_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_2_2 = nn.BatchNorm2d(64)
        self.conv_2_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_2_3 = nn.BatchNorm2d(64)
        # T3
        self.conv_3_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_3_3 = nn.BatchNorm2d(64)
        self.conv_3_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_3_4 = nn.BatchNorm2d(64)
        # T4
        self.conv_4_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_4_4 = nn.BatchNorm2d(64)
        self.conv_4_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_4_5 = nn.BatchNorm2d(64)
        # T5
        self.conv_5_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_5_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_21 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_21 = nn.BatchNorm2d(64)
        self.conv_31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_31 = nn.BatchNorm2d(64)
        self.conv_41 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_41 = nn.BatchNorm2d(64)
        self.conv_51 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_51 = nn.BatchNorm2d(64)

        # T21
        self.conv_21_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_21_3 = nn.BatchNorm2d(64)
        # T31
        self.conv_31_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_31_3 = nn.BatchNorm2d(64)
        self.conv_31_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_31_4 = nn.BatchNorm2d(64)
        # T41
        self.conv_41_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_41_4 = nn.BatchNorm2d(64)
        self.conv_41_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_41_5 = nn.BatchNorm2d(64)
        # T51
        self.conv_51_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_51_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_32 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_32 = nn.BatchNorm2d(64)
        self.conv_42 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_42 = nn.BatchNorm2d(64)
        self.conv_52 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_52 = nn.BatchNorm2d(64)

        # T32
        self.conv_32_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_32_4 = nn.BatchNorm2d(64)
        # T42
        self.conv_42_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_42_4 = nn.BatchNorm2d(64)
        self.conv_42_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_42_5 = nn.BatchNorm2d(64)
        # T52
        self.conv_52_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_52_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_43 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_43 = nn.BatchNorm2d(64)
        self.conv_53 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_53 = nn.BatchNorm2d(64)

        # T43
        self.conv_43_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_43_5 = nn.BatchNorm2d(64)
        # T53
        self.conv_53_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_53_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_54 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_54 = nn.BatchNorm2d(64)


    def forward(self, in_feat):
        # T1
        out_1_2 = F.interpolate(in_feat[0], size=in_feat[1].size()[2:], mode='bilinear')
        out_1_2 = F.relu(self.bn_1_2(self.conv_1_2(out_1_2)), inplace=True)
        # T2
        out_2_2 = in_feat[1]
        out_2_2 = F.relu(self.bn_2_2(self.conv_2_2(out_2_2)), inplace=True)
        out_2_3 = F.interpolate(in_feat[1], size=in_feat[2].size()[2:], mode='bilinear')
        out_2_3 = F.relu(self.bn_2_3(self.conv_2_3(out_2_3)), inplace=True)
        # T3
        out_3_3 = in_feat[2]
        out_3_3 = F.relu(self.bn_3_3(self.conv_3_3(out_3_3)), inplace=True)
        out_3_4 = F.interpolate(in_feat[2], size=in_feat[3].size()[2:], mode='bilinear')
        out_3_4 = F.relu(self.bn_3_4(self.conv_3_4(out_3_4)), inplace=True)
        # T4
        out_4_4 = in_feat[3]
        out_4_4 = F.relu(self.bn_4_4(self.conv_4_4(out_4_4)), inplace=True)
        out_4_5 = F.interpolate(in_feat[3], size=in_feat[4].size()[2:], mode='bilinear')
        out_4_5 = F.relu(self.bn_4_5(self.conv_4_5(out_4_5)), inplace=True)
        # T5
        out_5_5 = in_feat[4]
        out_5_5 = F.relu(self.bn_5_5(self.conv_5_5(out_5_5)), inplace=True)
        # sum
        out_21 = out_1_2 + out_2_2
        out_21 = F.relu(self.bn_21(self.conv_21(out_21)), inplace=True)
        out_31 = out_2_3 + out_3_3
        out_31 = F.relu(self.bn_31(self.conv_31(out_31)), inplace=True)
        out_41 = out_3_4 + out_4_4
        out_41 = F.relu(self.bn_41(self.conv_41(out_41)), inplace=True)
        out_51 = out_4_5 + out_5_5
        out_51 = F.relu(self.bn_51(self.conv_51(out_51)), inplace=True)

        # T21
        out_21_3 = F.interpolate(out_21, size=in_feat[2].size()[2:], mode='bilinear')
        out_21_3 = F.relu(self.bn_21_3(self.conv_21_3(out_21_3)), inplace=True)
        # T31
        out_31_3 = out_31
        out_31_3 = F.relu(self.bn_31_3(self.conv_31_3(out_31_3)), inplace=True)
        out_31_4 = F.interpolate(out_31, size=in_feat[3].size()[2:], mode='bilinear')
        out_31_4 = F.relu(self.bn_31_4(self.conv_31_4(out_31_4)), inplace=True)
        # T41
        out_41_4 = out_41
        out_41_4 = F.relu(self.bn_41_4(self.conv_41_4(out_41_4)), inplace=True)
        out_41_5 = F.interpolate(out_41, size=in_feat[4].size()[2:], mode='bilinear')
        out_41_5 = F.relu(self.bn_41_5(self.conv_41_5(out_41_5)), inplace=True)
        # T51
        out_51_5 = out_51
        out_51_5 = F.relu(self.bn_51_5(self.conv_51_5(out_51_5)), inplace=True)
        # sum
        out_32 = out_21_3 + out_31_3
        out_32 = F.relu(self.bn_32(self.conv_32(out_32)), inplace=True)
        out_42 = out_31_4 + out_41_4
        out_42 = F.relu(self.bn_42(self.conv_42(out_42)), inplace=True)
        out_52 = out_41_5 + out_51_5
        out_52 = F.relu(self.bn_52(self.conv_52(out_52)), inplace=True)

        # T32
        out_32_4 = F.interpolate(out_32, size=in_feat[3].size()[2:], mode='bilinear')
        out_32_4 = F.relu(self.bn_32_4(self.conv_32_4(out_32_4)), inplace=True)
        # T42
        out_42_4 = out_42
        out_42_4 = F.relu(self.bn_42_4(self.conv_42_4(out_42_4)), inplace=True)
        out_42_5 = F.interpolate(out_42, size=in_feat[4].size()[2:], mode='bilinear')
        out_42_5 = F.relu(self.bn_42_5(self.conv_42_5(out_42_5)), inplace=True)
        # T52
        out_52_5 = out_52
        out_52_5 = F.relu(self.bn_52_5(self.conv_52_5(out_52_5)), inplace=True)
        # sum
        out_43 = out_32_4 + out_42_4
        out_43 = F.relu(self.bn_43(self.conv_43(out_43)), inplace=True)  #
        out_53 = out_52_5 + out_42_5
        out_53 = F.relu(self.bn_53(self.conv_53(out_53)), inplace=True)  #

        # T43
        out_43_5 = F.interpolate(out_43, size=in_feat[4].size()[2:], mode='bilinear')
        out_43_5 = F.relu(self.bn_43_5(self.conv_43_5(out_43_5)), inplace=True)
        # T53
        out_53_5 = out_53
        out_53_5 = F.relu(self.bn_53_5(self.conv_53_5(out_53_5)), inplace=True)
        # sum
        out_54 = out_43_5 + out_53_5
        out_54 = F.relu(self.bn_54(self.conv_54(out_54)), inplace=True)  #

        return out_32, out_43, out_54

    def initialize(self):
        weight_init(self)


class StrucDe(nn.Module):
    def __init__(self):
        super(StrucDe, self).__init__()

        self.conv0 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv4_reduce = nn.Conv2d(32, 16, kernel_size=1)
        self.bn4_reduce = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

    def forward(self, in_feat):
        out0 = F.relu(self.bn0(self.conv0(in_feat[0])), inplace=True)
        out0_up = F.interpolate(out0, size=in_feat[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(out0_up + in_feat[1])), inplace=True)
        out1_up = F.interpolate(out1, size=in_feat[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(out1_up + in_feat[2])), inplace=True)
        out2_up = F.interpolate(out2, size=in_feat[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(out2_up + in_feat[3])), inplace=True)
        out3_up = F.interpolate(out3, size=in_feat[4].size()[2:], mode='bilinear')
        out4 = F.relu(self.bn4(self.conv4(out3_up + in_feat[4])), inplace=True)
        out4_up = F.interpolate(out4, size=in_feat[5].size()[2:], mode='bilinear')
        out4_up = F.relu(self.bn4_reduce(self.conv4_reduce(out4_up )), inplace=True)
        out5 = F.relu(self.bn5(self.conv5(out4_up + in_feat[5])), inplace=True)

        return out0, out1, out2, out3, out4, out5

    def initialize(self):
        weight_init(self)


class DCM(nn.Module):
    def __init__(self):
        super(DCM, self).__init__()

        self.convLR0 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnLR0 = nn.BatchNorm2d(32)

        self.convLR1 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnLR1 = nn.BatchNorm2d(32)

        self.convLR2 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnLR2 = nn.BatchNorm2d(32)

        self.convLR3 = nn.Conv2d(32, 32, kernel_size=1)
        self.bnLR3 = nn.BatchNorm2d(32)

    def forward(self, featLR, featHR):

        temp = F.relu(self.bnLR0(self.convLR0(featLR[0])), inplace=True)
        featHR0 = featHR[0] - temp
        temp = F.relu(self.bnLR1(self.convLR1(featLR[1])), inplace=True)
        featHR1 = featHR[1] - temp
        temp = F.relu(self.bnLR2(self.convLR2(featLR[2])), inplace=True)
        featHR2 = featHR[2] - temp

        return featHR0, featHR1, featHR2

    def initialize(self):
        weight_init(self)


class UDUN(nn.Module):
    def __init__(self, channels=64):
        super(UDUN, self).__init__()

        self.convHR0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1), nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.convHR1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.convHR2 = nn.Sequential(nn.Conv2d(256, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.convHR3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convHR4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convHR5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.convLR1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.convLR2 = nn.Sequential(nn.Conv2d(256, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.convLR3 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1), nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.convLR4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.convLR5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.dcm = DCM()
        self.trunk = TrunkDe()
        self.struct = StrucDe()
        self.union_de = UnionDe()

        self.linear_t = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear_s = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=3, padding=1))

        #weight_init(self)

        self.bkbone = resnet50()

    def forward(self, x):
        print(x.shape)
        y = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        outHR0 = self.convHR0(x)

        outHR1, outHR2, outHR3, outHR4, outHR5 = self.bkbone(x)
        outLR1, outLR2, outLR3, outLR4, outLR5 = self.bkbone(y)

        outHR1, outHR2, outHR3, outHR4, outHR5 = self.convHR1(outHR1), self.convHR2(outHR2), self.convHR3(
            outHR3), self.convHR4(outHR4), self.convHR5(outHR5)
        outLR1, outLR2, outLR3, outLR4, outLR5 = self.convLR1(outLR1), self.convLR2(outLR2), self.convLR3(
            outLR3), self.convLR4(outLR4), self.convLR5(outLR5)

        out_T32, out_T43, out_T54 = self.trunk([outLR5, outLR4, outHR5, outHR4, outHR3])
        outLR3, outLR2, outLR1 = self.dcm([out_T32, out_T43, out_T54], [outLR3, outLR2, outLR1, outHR2])
        out_S1, out_S2, out_S3, out_S4, out_S5, out_S6 = self.struct([outLR3, outLR2, outLR1, outHR2, outHR1, outHR0])
        maskFeature = self.union_de([out_T32, out_T43, out_T54], [out_S1, out_S2, out_S3, out_S4, out_S5, out_S6])

        out_mask = self.linear(maskFeature)
        out_trunk = self.linear_t(out_T54)
        out_struct = self.linear_s(out_S6)

        return out_trunk, out_struct, out_mask


if __name__ == '__main__':
    net = UDUN().eval().cuda()
    x = torch.rand(2, 3, 1024,1024).cuda()
    y = net(x)
    print([i.shape for i in y])