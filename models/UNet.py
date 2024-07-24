import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
# from losses import structure_loss, NCLoss


class UPM(nn.Module):
    def __init__(self, in_channels=64):
        super(UPM, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, 1, 3, 1, 1)

    def forward(self, x):
        pre = self.final_conv(x).sigmoid()
        uncentary_region = 1 - (2 * pre - 1).abs().pow(2)

        return x


class UNetResNet50(nn.Module):
    def __init__(self, n_classes=1):
        super(UNetResNet50, self).__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)

        self.enc1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # Output: 64x64x64

        self.enc2 = resnet.layer1  # Output: 256x64x64
        self.enc3 = resnet.layer2  # Output: 512x32x32
        self.enc4 = resnet.layer3  # Output: 1024x16x16
        self.enc5 = resnet.layer4  # Output: 2048x8x8

        self.center = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )  # Center block: 1024x8x8 

        self.dec5 = self.decoder_block(2048 + 1024, 1024)  # Output: 1024x16x16
        self.dec4 = self.decoder_block(1024 + 1024, 512)  # Output: 512x32x32
        self.dec3 = self.decoder_block(512 + 512, 256)  # Output: 256x64x64
        self.dec2 = self.decoder_block(256 + 256, 64)  # Output: 64x128x128
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)  # Output: n_classesx128x128

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, data):
        x = data['image']
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        center = self.center(enc5)

        dec5 = self.dec5(torch.cat([center, enc5], dim=1))
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        out2 = F.interpolate(self.final(dec2), scale_factor=2, mode='bilinear')
        return {'res': [out2]}  

   
if __name__ == "__main__":
    model = UNetResNet50()
    x = torch.randn(3, 3, 256, 256)
    boxes = torch.tensor([[50, 50], [51, 51]])
    gts = torch.randn(3, 1, 256, 256)
    out = model(x, boxes, gts)
    print(out.shape)