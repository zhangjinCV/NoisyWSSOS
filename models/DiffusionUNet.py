import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization(channels):
    return nn.GroupNorm(32, channels)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class QKVAttention(nn.Module):
    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / (ch ** 0.25)
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight, dim=-1)
        return torch.einsum("bts,bcs->bct", weight, v)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(Downsample(64, use_conv=True), DoubleConv(64, 128))
        self.down2 = nn.Sequential(Downsample(128, use_conv=True), DoubleConv(128, 256))
        self.down3 = nn.Sequential(Downsample(256, use_conv=True), DoubleConv(256, 512))
        self.down4 = nn.Sequential(Downsample(512, use_conv=True), DoubleConv(512, 512))

        self.middle_block = nn.Sequential(
            DoubleConv(512, 512),
            AttentionBlock(512),
            DoubleConv(512, 512)
        )

        self.up1 = nn.Sequential(Upsample(1024, use_conv=True), DoubleConv(1024, 256))
        self.up2 = nn.Sequential(Upsample(512, use_conv=True), DoubleConv(512, 128))
        self.up3 = nn.Sequential(Upsample(256, use_conv=True), DoubleConv(256, 64))
        self.up4 = nn.Sequential(Upsample(128, use_conv=True), DoubleConv(128, 64))
        self.outc = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.attention1 = AttentionBlock(256)
        self.attention2 = AttentionBlock(128)

    def forward(self, data):
        x1 = data['image']
        x2 = data.get('bbox_image')
        x = torch.cat([x1, x2], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.middle_block(x5)

        x = self.up1(torch.cat([x, F.interpolate(x4, x.size()[2:])], dim=1))
        x = self.attention1(x)

        x = self.up2(torch.cat([x, F.interpolate(x3, x.size()[2:])], dim=1))
        x = self.attention2(x)

        x = self.up3(torch.cat([x, F.interpolate(x2, x.size()[2:])], dim=1))
        x = self.up4(torch.cat([x, F.interpolate(x1, x.size()[2:])], dim=1))

        logits = self.outc(x)
        res = {"res": [logits]}
        return res

def main():
    import torch
    from torchsummary import summary
    from thop import profile, clever_format

    n_channels = 6  # 因为我们将两个3通道的图像连接起来
    n_classes = 1

    model = UNet(n_channels, n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model Summary:")
    summary(model, [(3, 256, 256), (3, 256, 256)])

    x1 = torch.randn(1, 3, 256, 256).to(device)
    x2 = torch.randn(1, 3, 256, 256).to(device)
    flops, params = profile(model, inputs=(x1, x2))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

if __name__ == "__main__":
    main()
