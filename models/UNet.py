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


import torch
import torch.nn as nn

class ResBlock(nn.Sequential):
    def __init__(self, num_channels, kernel_size, norm_layer):
        super(ResBlock, self).__init__()
        layers = []
        layers += [
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(num_channels, num_channels, kernel_size, bias=False),
            norm_layer(num_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(num_channels, num_channels, kernel_size, bias=False),
            norm_layer(num_channels, affine=True),
        ]        
        self.blocks = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.blocks(x) + x


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=8, num_channels=32, kernel_size=3,  norm_layer=nn.BatchNorm2d):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.self_attention4 = SelfAttention(512)
        self.self_attention5 = SelfAttention(1024)
        
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, data):
        x = data['image']
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
        
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        bottleneck = self.self_attention5(bottleneck)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.self_attention4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.conv_final(dec1)
        res = {'res': [out]}
        return res
   
if __name__ == "__main__":
    model = UNetResNet50()
    x = torch.randn(3, 3, 256, 256)
    boxes = torch.tensor([[50, 50], [51, 51]])
    gts = torch.randn(3, 1, 256, 256)
    out = model(x, boxes, gts)
    print(out.shape)