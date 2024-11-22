import abc
import logging
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import pvt_v2_b4
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops import rearrange


def rescale_2x(x: torch.Tensor, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def global_avgpool(x: torch.Tensor):
    return x.mean((-1, -2), keepdim=True)


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError


class ConvBN(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, g=1, bias=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        return self.bn(self.conv(x))


class CBR(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gn_groups=8,
        bias=False,
        act_name="relu",
        inplace=True,
    ):
        """
        执行流程Conv2d => GroupNormalization [=> Activation]

        Args:
            in_planes: 模块输入通道数
            out_planes: 模块输出通道数
            kernel_size: 内部卷积操作的卷积核大小
            stride: 卷积步长
            padding: 卷积padding
            dilation: 卷积的扩张率
            groups: 卷积分组数，需满足pytorch自身要求
            gn_groups: GroupNormalization的分组数，默认为4
            bias: 是否启用卷积的偏置，默认为False
            act_name: 使用的激活函数，默认为relu，设置为None的时候则不使用激活函数
            inplace: 设置激活函数的inplace参数
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="gn", module=nn.GroupNorm(num_groups=gn_groups, num_channels=out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=inplace))


class PixelNormalizer(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

        Args:
            mean (tuple, optional): the mean value. Defaults to (0.485, 0.456, 0.406).
            std (tuple, optional): the std value. Defaults to (0.229, 0.224, 0.225).
        """
        super().__init__()
        # self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        self.register_buffer(name="mean", tensor=torch.Tensor(mean).reshape(3, 1, 1))
        self.register_buffer(name="std", tensor=torch.Tensor(std).reshape(3, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean.flatten()}, std={self.std.flatten()})"

    def forward(self, x):
        """normalize x by the mean and std values

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        Albumentations:

        ```
            mean = np.array(mean, dtype=np.float32)
            mean *= max_pixel_value
            std = np.array(std, dtype=np.float32)
            std *= max_pixel_value
            denominator = np.reciprocal(std, dtype=np.float32)

            img = img.astype(np.float32)
            img -= mean
            img *= denominator
        ```
        """
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        """A simple ASPP variant.

        Args:
            in_dim (int): Input channels.
            out_dim (int): Output channels.
            dilation (int, optional): Dilation of the convolution operation. Defaults to 3.
        """
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1))

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)

        # dilation branch
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)

        # global branch
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))


class DifferenceAwareOps(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

        self.temperal_proj_norm = nn.LayerNorm(num_frames, elementwise_affine=False)
        self.temperal_proj_kv = nn.Linear(num_frames, 2 * num_frames, bias=False)
        self.temperal_proj = nn.Sequential(
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
        )
        for t in self.parameters():
            nn.init.zeros_(t)

    def forward(self, x):
        if self.num_frames == 1:
            return x

        unshifted_x_tmp = rearrange(x, "(b t) c h w -> b c h w t", t=self.num_frames)
        B, C, H, W, T = unshifted_x_tmp.shape
        shifted_x_tmp = torch.roll(unshifted_x_tmp, shifts=1, dims=-1)
        diff_q = shifted_x_tmp - unshifted_x_tmp  # B,C,H,W,T
        diff_q = self.temperal_proj_norm(diff_q)  # normalization along the time

        # merge all channels
        diff_k, diff_v = self.temperal_proj_kv(diff_q).chunk(2, dim=-1)
        diff_qk = torch.einsum("bxhwt, byhwt -> bxyt", diff_q, diff_k) * (H * W) ** -0.5
        temperal_diff = torch.einsum("bxyt, byhwt -> bxhwt", diff_qk.softmax(dim=2), diff_v)

        temperal_diff = rearrange(temperal_diff, "b c h w t -> (b c) t h w")
        shifted_x_tmp = self.temperal_proj(temperal_diff)  # combine different time step
        shifted_x_tmp = rearrange(shifted_x_tmp, "(b c) t h w -> (b t) c h w", c=x.shape[1])
        return x + shifted_x_tmp


class RGPU(nn.Module):
    def __init__(self, in_c, num_groups=6, hidden_dim=None, num_frames=1):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(
            DifferenceAwareOps(num_frames=num_frames),
            ConvBNReLU(num_groups * hidden_dim, in_c, 3, 1, 1, act_name=None),
        )
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []
        gates = []

        group_id = 0
        curr_x = xs[group_id]
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        for group_id in range(1, self.num_groups - 1):
            curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
            branch_out = self.interact[str(group_id)](curr_x)
            curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)

        group_id = self.num_groups - 1
        curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_gate = branch_out.chunk(2, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        out = torch.cat(outs, dim=1)
        gate = self.gate_genator(torch.cat(gates, dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch

        self.conv_lms = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)  # inter-branch
        self.initial_merge = ConvBNReLU(3 * in_dim, 3 * in_dim, 1)  # inter-branch

        self.num_groups = num_groups
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, l, m, s):
        tgt_size = m.shape[2:]

        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=m.shape[2:])

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)
        lms = torch.cat([l, m, s], dim=1)  # BT,3C,H,W

        attn = self.conv_lms(lms)  # BT,3C,H,W
        attn = rearrange(attn, "bt (nb ng d) h w -> (bt ng) (nb d) h w", nb=3, ng=self.num_groups)
        attn = self.trans(attn)  # BTG,3,H,W
        attn = attn.unsqueeze(dim=2)  # BTG,3,1,H,W

        x = self.initial_merge(lms)
        x = rearrange(x, "bt (nb ng d) h w -> (bt ng) nb d h w", nb=3, ng=self.num_groups)
        x = (attn * x).sum(dim=1)
        x = rearrange(x, "(bt ng) d h w -> bt (ng d) h w", ng=self.num_groups)
        return x
    

class LayerNorm2d(nn.Module):
    """
    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class _ZoomNeXt_Base(nn.Module):
    @abc.abstractmethod
    def body(self):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        logits = self.body(data=data)
        res = {'res': [logits]}
        return res

class PvtV2B2_ZoomNeXt(_ZoomNeXt_Base):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

        self.embed_dims = self.encoder.embed_dims
        self.tra_5 = SimpleASPP(self.embed_dims[3], out_dim=mid_dim)
        self.siu_5 = MHSIU(mid_dim, siu_groups)
        self.hmu_5 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_4 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.siu_4 = MHSIU(mid_dim, siu_groups)
        self.hmu_4 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_3 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.siu_3 = MHSIU(mid_dim, siu_groups)
        self.hmu_3 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_2 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.siu_2 = MHSIU(mid_dim, siu_groups)
        self.hmu_2 = RGPU(mid_dim, hmu_groups, num_frames=num_frames)

        self.tra_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_b4()

    def normalize_encoder(self, x):
        features = self.encoder(x)
        return features

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        l, m, s = self.tra_5(l_trans_feats[3]), self.tra_5(m_trans_feats[3]), self.tra_5(s_trans_feats[3])
        lms = self.siu_5(l=l, m=m, s=s)
        x = self.hmu_5(lms)

        l, m, s = self.tra_4(l_trans_feats[2]), self.tra_4(m_trans_feats[2]), self.tra_4(s_trans_feats[2])
        lms = self.siu_4(l=l, m=m, s=s)
        x = self.hmu_4(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_3(l_trans_feats[1]), self.tra_3(m_trans_feats[1]), self.tra_3(s_trans_feats[1])
        lms = self.siu_3(l=l, m=m, s=s)
        x = self.hmu_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m, s = self.tra_2(l_trans_feats[0]), self.tra_2(m_trans_feats[0]), self.tra_2(s_trans_feats[0])
        lms = self.siu_2(l=l, m=m, s=s)
        x = self.hmu_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        x = self.tra_1(x)
        return self.predictor(x)


if __name__ == '__main__':
    model = PvtV2B2_ZoomNeXt()
    x = torch.rand(2, 3, 384, 384)
    y = torch.rand(2, 3, 224, 224)
    z = torch.rand(2, 3, 512, 512)
    data = {"image_s":y, 'image_m':x, 'image_l':z}
    y = model(data)
    print(y.shape)