# Channel Separated Convolutional Network (CSN) as presented in Video Classification with Channel-Separated Convolutional Networks(https://arxiv.org/pdf/1904.02811v4.pdf)
# replace 3x3x3 convolution with 1x1x1 conv + 3x3x3 depthwise convolution (ip) or with 3x3x3 depthwise convolution (ir)

import torch
import torch.nn as nn
import copy
import numpy as np
from network.non_local_dot_product import NONLocalBlock3D

class TATLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        self.attention_conv1 = nn.Conv3d(self.in_channels*2, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)) 
        self.gn1 = nn.GroupNorm(4,128)
        self.attention_conv2 = nn.Conv3d(128, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1)) 
        self.gn2 = nn.GroupNorm(1,1)
        self.downsample_4x = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.downsample_global = nn.MaxPool3d(kernel_size=(1, 7, 7), stride=1)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        batch_size = input.shape[0]
        in_channels = input.shape[1]
        clip_len = input.shape[2]
        sz = input.shape[3]
        
        input_new = torch.zeros(batch_size, in_channels*2, clip_len, sz, sz).cuda()
        attention_map = torch.zeros(batch_size, clip_len).cuda()
        for i in range(clip_len):
            input_new[:,:,i,:,:] = torch.cat( (input[:,:,int(clip_len/2),:,:], input[:,:,i,:,:]) , 1)

        out = self.relu(self.gn1(self.attention_conv1(input_new)))
        out = self.downsample_4x(out)
        out = self.relu(self.gn2(self.attention_conv2(out)))
        out = self.downsample_global(out) # [Batch, 1, T, H, W]

        attention_map = out.view(batch_size,clip_len)

        return self.softmax(attention_map)


        
class CSNBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, channels, stride=1, mode='ip'):
        super().__init__()
        
        assert mode in ['ip', 'ir']
        self.mode = mode
        
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.gn1 = nn.GroupNorm(int(channels/16),channels)
        self.relu = nn.ReLU(inplace=True)
        
        conv2 = []
        if self.mode == 'ip':
            conv2.append(nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False))
        conv2.append(nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=channels))
        self.conv2 = nn.Sequential(*conv2)
        self.bn2 = nn.BatchNorm3d(channels)
        self.gn2 = nn.GroupNorm(int(channels/16),channels)
        
        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)
        self.gn3 = nn.GroupNorm(int(channels * self.expansion / 16), channels * self.expansion)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm3d(channels * self.expansion)
                nn.GroupNorm(int(channels * self.expansion / 16), channels * self.expansion)
            )
        
    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.gn3(out)
            
        out += shortcut
        out = self.relu(out)
        
        return out


class CSN(nn.Module):
    def __init__(self, block, layers, num_classes, mode='ip', add_landmarks=False):
        super().__init__()
        
        assert mode in ['ip', 'ir']
        self.mode = mode
        self.add_landmarks = add_landmarks
        
        self.in_channels = 64
        if add_landmarks:
            self.conv1 = nn.Conv3d(4, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        else:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.gn1 = nn.GroupNorm(1, 64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        #self.tat_layer = TATLayer(in_channels=256)
        self.nl_1 = NONLocalBlock3D(in_channels=256)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.nl_2 = NONLocalBlock3D(in_channels=512)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.nl_3 = NONLocalBlock3D(in_channels=1024)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1) # 原文是avgpool
        self.global_max_pool = nn.AdaptiveMaxPool3d(1) 
        self.fc1 = nn.Linear(512 * block.expansion * 2, 9)
        #self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride, mode=self.mode))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels, mode=self.mode))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, debug=False):
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        if debug:
            print("shape1:", out.shape)
        out = self.max_pool(out)
        if debug:
            print("shape2:", out.shape)
        
        out = self.layer1(out)
        #out = self.nl_1(out)

        # 添加attention layer
        # attention_map = self.tat_layer(out)

        # #print(attention_map[0,:])
        # for b in range(out.shape[0]):
        #     for i in range(out.shape[2]):
        #         out[b,:,i,:,:] *= attention_map[b,i]

        if debug:
            print("shape3:", out.shape)
        
        out = self.layer2(out)
        out = self.nl_2(out)
        if debug:
            print("shape4:", out.shape)
        out = self.layer3(out)
        out = self.nl_3(out)
        if debug:
            print("shape5:", out.shape)
        out = self.layer4(out)
        if debug:
            print("shape6:", out.shape)
        
        out1 = self.global_max_pool(out)
        out2 = self.global_max_pool(out)
        if debug:
            print("shape7:", out1.shape)
        
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc1(out)
        #out = self.relu(out)
        #out = self.dropout(out)
        #out = self.fc2(out)
        #out = self.tanh(out)
        
        return self.tanh(out[:, 0:2]), out[:, 2:9]


def csn26(num_classes, mode='ip', add_landmarks=False):
    return CSN(CSNBottleneck, [2,2,2,1], num_classes=num_classes, mode=mode, add_landmarks=add_landmarks)
    # 1241


def csn50(num_classes, mode='ip', add_landmarks=False):
    return CSN(CSNBottleneck, [3,4,6,3], num_classes=num_classes, mode=mode, add_landmarks=add_landmarks)


def csn101(num_classes, mode='ip', add_landmarks=False):
    return CSN(CSNBottleneck, [3,4,23,3], num_classes=num_classes, mode=mode, add_landmarks=add_landmarks)


def csn152(num_classes, add_landmarks=False):
    return CSN(CSNBottleneck, [3,8,36,3], num_classes=num_classes, add_landmarks=add_landmarks)
    