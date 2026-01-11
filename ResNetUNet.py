import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    """双层卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, local_model=False):
        super().__init__()
        # 加载预训练ResNet34（ImageNet权重）
        if not local_model:
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet34(weights=None)

        # 冻结前两层权重（可选，小数据集建议冻结）
        for param in list(self.resnet.parameters())[:10]:
            param.requires_grad = False

        # 编码器（复用ResNet的特征提取层）
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)  # 64, H/2, W/2 (256->128)
        self.encoder2 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)  # 64, H/4, W/4 (128->64)
        self.encoder3 = self.resnet.layer2  # 128, H/8, W/8 (64->32)
        self.encoder4 = self.resnet.layer3  # 256, H/16, W/16 (32->16)
        self.encoder5 = self.resnet.layer4  # 512, H/32, W/32 (16->8)

        # 解码器（转置卷积上采样）
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 512->256, 8->16
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 256->128, 16->32
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128->64, 32->64
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 64->64, 64->128
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 新增：64->64, 128->256（匹配掩码尺寸）

        # 融合卷积块（拼接后降维）
        self.decoder1 = DoubleConv(256 + 256, 256)  # e4(256) + up1(256)
        self.decoder2 = DoubleConv(128 + 128, 128)  # e3(128) + up2(128)
        self.decoder3 = DoubleConv(64 + 64, 64)  # e2(64) + up3(64)
        self.decoder4 = DoubleConv(64 + 64, 64)  # e1(64) + up4(64)
        self.decoder5 = DoubleConv(64, 64)  # 新增：融合最后一次上采样结果

        # 输出层（1x1卷积）
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码阶段
        e1 = self.encoder1(x)  # [B, 64, 128, 128]
        e2 = self.encoder2(e1)  # [B, 64, 64, 64]
        e3 = self.encoder3(e2)  # [B, 128, 32, 32]
        e4 = self.encoder4(e3)  # [B, 256, 16, 16]
        e5 = self.encoder5(e4)  # [B, 512, 8, 8]

        # 解码阶段 + 跳跃连接
        d1 = self.up1(e5)  # [B, 256, 16, 16]
        d1 = torch.cat([d1, e4], dim=1)  # 拼接e4特征 [B, 512, 16, 16]
        d1 = self.decoder1(d1)  # [B, 256, 16, 16]

        d2 = self.up2(d1)  # [B, 128, 32, 32]
        d2 = torch.cat([d2, e3], dim=1)  # 拼接e3特征 [B, 256, 32, 32]
        d2 = self.decoder2(d2)  # [B, 128, 32, 32]

        d3 = self.up3(d2)  # [B, 64, 64, 64]
        d3 = torch.cat([d3, e2], dim=1)  # 拼接e2特征 [B, 128, 64, 64]
        d3 = self.decoder3(d3)  # [B, 64, 64, 64]

        d4 = self.up4(d3)  # [B, 64, 128, 128]
        d4 = torch.cat([d4, e1], dim=1)  # 拼接e1特征 [B, 128, 128, 128]
        d4 = self.decoder4(d4)  # [B, 64, 128, 128]

        # 新增：最后一次上采样，放大到256×256
        d5 = self.up5(d4)  # [B, 64, 256, 256]
        d5 = self.decoder5(d5)  # [B, 64, 256, 256]

        # 输出（二分类用sigmoid）
        out = self.out(d5)  # [B, 1, 256, 256]（与掩码尺寸一致）
        return torch.sigmoid(out)
