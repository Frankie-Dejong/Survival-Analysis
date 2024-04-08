import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channel, down_sample):
        '''
        down_sample = 1 for not down sampling
        down_sample = 2 for down sampling 2 times
        '''
        assert down_sample == 1 or down_sample == 2
        super(DownBlock, self).__init__()
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channel, in_channel * down_sample, kernel_size=3, stride=down_sample, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * down_sample)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * down_sample, in_channel * down_sample, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * down_sample)
        if self.down_sample == 2:
            self.down_sample_block = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel * down_sample, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(in_channel * down_sample)
                )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample == 2:
            identity = self.down_sample_block(x)
        out = out + identity
        out = self.relu(out)
        return out


'''
basically follow the structure of resnet18, but remove the last avg pooling & fc to get a feature map
'''
class Encoder(nn.Module):
    def __init__(self, out_channels, zero_init_residual=True):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
                DownBlock(64, 1),
                DownBlock(64, 1)
            )
        self.layer2 = nn.Sequential(
                DownBlock(64, 2),
                DownBlock(128, 1)
            )
        self.layer3 = nn.Sequential(
                DownBlock(128, 2),
                DownBlock(256, 1)
            )
        # self.layer4 = nn.Sequential(
        #         DownBlock(256, 2),
        #         DownBlock(512, 1)
        #     )
        # self.layer5 = nn.Sequential(
        #     DownBlock(512, 2),
        #     DownBlock(1024, 1)
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_conv = nn.Conv2d(256, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, DownBlock):
                        nn.init.constant_(m.bn2.weight, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.avgpool(x)
        return self.out_conv(x)



'''
I'm not sure about this design. I do not follow the design of unet, which concat the feature map of encoder with decoder because i wanna force the encoder to learn a high quality 
representation of image, so i decide not to allow the encoder see other feature maps. Then I use the same residual design as resnet, basically a resnet but  replace convs with conv_trans
'''
class UpBlock(nn.Module):
    def __init__(self, in_channels, down_feature=1, mode='bilinear', scale_factor=2):
        '''
        down_feature = 1 for keep the in_channels
        down_feature = 2 for down sample channels for 2 times
        '''
        assert down_feature == 1 or down_feature == 2
        super(UpBlock, self).__init__()
        if mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
        
        out_channels = in_channels // down_feature
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.down_feature = down_feature
        if down_feature == 2:
            self.down_feature_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        x = self.up(x)
        identity = x
        if self.down_feature == 2:
            identity = self.down_feature_block(identity)

        x = self.conv(x) + identity
        return self.relu(x)



class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.up1 = UpBlock(512, 1)
        self.up2 = UpBlock(512, 2)
        self.up3 = UpBlock(256, 2)
        self.up4 = UpBlock(128, 2)
        # self.up5 = UpBlock(64, 1)
        # self.up6 = UpBlock(64, 1)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)


    def forward(self, x):
        x = self.in_conv(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x = self.up5(x)
        # x = self.up6(x)
        return self.out_conv(x)


class EncDec(nn.Module):
    def __init__(self, hidden_channels):
        super(EncDec, self).__init__()
        self.enc = Encoder(hidden_channels)
        self.dec = Decoder(hidden_channels)
        
        
    def forward(self, x):
        h = self.enc(x)
        return self.dec(h)








