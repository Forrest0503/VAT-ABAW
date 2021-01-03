import math
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import torchvision_models
from .torchvision_models import load_pretrained, inflate_pretrained, modify_resnets
from network.non_local_gaussian import NONLocalBlock3D

class STABlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(STABlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # self.sa1 = nn.Conv3d(in_channels=self.in_channels, out_channels=128,
        #                  kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0) # 8, 14, 14
        # self.sa2 = nn.Conv3d(in_channels=128, out_channels=128,
        #                  kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0) # 8, 7, 7
        # self.sa3 = nn.Conv3d(in_channels=128, out_channels=128,
        #                  kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=0) # 128, 8, 1, 1
        # self.sa4 = nn.Conv3d(in_channels=128, out_channels=1,
        #                  kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0) # 1, 8, 1, 1

        self.Qs = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.Ks = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.Vs = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.Ws = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)

        self.Q = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)

        nn.init.constant_(self.Q.weight, 0)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.K.weight, 0)
        nn.init.constant_(self.K.bias, 0)
        nn.init.constant_(self.Qs.weight, 0)
        nn.init.constant_(self.Qs.bias, 0)
        nn.init.constant_(self.Ks.weight, 0)
        nn.init.constant_(self.Ks.bias, 0)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)   (b, 64, t=16/2, 28, 28)
        :return:
        '''

        batch_size = x.size(0)
        t = x.size(2)
        h = x.size(3)
        w = x.size(4)
        c = self.inter_channels

        # Single Attention
        # x_sa = self.sa1(x)
        # x_sa = self.sa2(x_sa)
        # x_sa = self.sa3(x_sa)
        # x_sa = self.sa4(x_sa).view(batch_size, -1) #[batch, t]
        # x_sa = F.sigmoid(x_sa)
        # # print(x_sa)
        # x_sa = x_sa.unsqueeze(2).unsqueeze(3).unsqueeze(4) #[batch, t, 1, 1, 1]
        # x = x.permute(0, 2, 1, 3, 4)
        
        # x = x.mul(x_sa)
        # x = x.permute(0, 2, 1, 3, 4)


        # Spatiao Attention
        # Query
        Q_x_s = self.Qs(x).view(batch_size, t, -1)
        Q_x_s = Q_x_s.permute(0, 2, 1) # [chw x t]

        # Key
        K_x_s = self.Ks(x).view(batch_size, t, -1) # [t x chw]

        # Value
        V_x_s = self.Vs(x).view(batch_size, t, -1)
        V_x_s = V_x_s.permute(0, 2, 1) # [chw x t]

        corr_s = torch.matmul(Q_x_s, K_x_s) # [chw x chw]
        corr_s_div_C = F.softmax(corr_s / math.sqrt(t), dim=-1)

        ys = torch.matmul(corr_s_div_C, V_x_s)
        ys = ys.permute(0, 2, 1).contiguous()
        ys = ys.view(batch_size, self.inter_channels, *x.size()[2:])


        # Temporal Attention
        # Query
        Q_x_t = self.Q(x).view(batch_size, c*h*w, -1)
        Q_x_t = Q_x_t.permute(0, 2, 1) # [t x chw]

        # Key
        K_x_t = self.K(x).view(batch_size, c*h*w, -1) # [chw x t]

        # Value
        V_x_t = self.V(x).view(batch_size, c*h*w, -1)
        V_x_t = V_x_t.permute(0, 2, 1) # [t x chw]

        corr_t = torch.matmul(Q_x_t, K_x_t) # [t x t]
        corr_t_div_C = F.softmax(corr_t / math.sqrt(t), dim=-1)

        yt = torch.matmul(corr_t_div_C, V_x_t)
        yt = yt.permute(0, 2, 1).contiguous()
        yt = yt.view(batch_size, self.inter_channels, *x.size()[2:])

        y_combine = ys + yt

        W = self.W(y_combine)
        z = W + x

        return z
    
class TABlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(TABlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        # self.sa1 = nn.Conv3d(in_channels=self.in_channels, out_channels=128,
        #                  kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0) # 8, 14, 14
        # self.sa2 = nn.Conv3d(in_channels=128, out_channels=128,
        #                  kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0) # 8, 7, 7
        # self.sa3 = nn.Conv3d(in_channels=128, out_channels=128,
        #                  kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=0) # 128, 8, 1, 1
        # self.sa4 = nn.Conv3d(in_channels=128, out_channels=1,
        #                  kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0) # 1, 8, 1, 1


        self.Q = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)

        nn.init.constant_(self.Q.weight, 0)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.K.weight, 0)
        nn.init.constant_(self.K.bias, 0)


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)   (b, 64, t=16/2, 28, 28)
        :return:
        '''

        batch_size = x.size(0)
        t = x.size(2)
        h = x.size(3)
        w = x.size(4)
        c = self.inter_channels

        # Single Attention
        # x_sa = self.sa1(x)
        # x_sa = self.sa2(x_sa)
        # x_sa = self.sa3(x_sa)
        # x_sa = self.sa4(x_sa).view(batch_size, -1) #[batch, t]
        # x_sa = F.sigmoid(x_sa)
        # # print(x_sa)
        # x_sa = x_sa.unsqueeze(2).unsqueeze(3).unsqueeze(4) #[batch, t, 1, 1, 1]
        # x = x.permute(0, 2, 1, 3, 4)
        
        # x = x.mul(x_sa)
        # x = x.permute(0, 2, 1, 3, 4)


        # Temporal Attention
        # Query
        Q_x_t = self.Q(x).view(batch_size, c*h*w, -1)
        Q_x_t = Q_x_t.permute(0, 2, 1) # [t x chw]

        # Key
        K_x_t = self.K(x).view(batch_size, c*h*w, -1) # [chw x t]

        # Value
        V_x_t = self.V(x).view(batch_size, c*h*w, -1)
        V_x_t = V_x_t.permute(0, 2, 1) # [t x chw]

        corr_t = torch.matmul(Q_x_t, K_x_t) # [t x t]
        corr_t_div_C = F.softmax(corr_t / math.sqrt(h*w*c), dim=-1)

        yt = torch.matmul(corr_t_div_C, V_x_t)
        yt = yt.permute(0, 2, 1).contiguous()
        yt = yt.view(batch_size, self.inter_channels, *x.size()[2:])

        W = self.W(yt)
        z = W + x

        return z


__all__ = [
    'ResNet3D', 'resnet3d10', 'resnet3d18', 'resnet3d34',
    'resnet3d50', 'resnet3d101', 'resnet3d152', 'resnet3d200',
]

model_urls = {
    'kinetics-400': defaultdict(lambda: None, {
        'resnet3d18': 'http://pretorched-x.csail.mit.edu/models/resnet3d18_kinetics-e9f44270.pth',
        'resnet3d34': 'http://pretorched-x.csail.mit.edu/models/resnet3d34_kinetics-7fed38dd.pth',
        'resnet3d50': 'http://pretorched-x.csail.mit.edu/models/resnet3d50_kinetics-aad059c9.pth',
        'resnet3d101': 'http://pretorched-x.csail.mit.edu/models/resnet3d101_kinetics-8d4c9d63.pth',
        'resnet3d152': 'http://pretorched-x.csail.mit.edu/models/resnet3d152_kinetics-575c47e2.pth',
    }),
    'moments': defaultdict(lambda: None, {
        'resnet3d50': 'http://pretorched-x.csail.mit.edu/models/resnet3d50_16seg_moments-6eb53860.pth',
    }),
}

num_classes = {'kinetics-400': 400, 'moments': 339}

pretrained_settings = defaultdict(dict)
input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in __all__:
    if model_name in ['ResNet3D']:
        continue
    for dataset, urls in model_urls.items():
        pretrained_settings[model_name][dataset] = {
            'input_space': 'RGB',
            'input_range': [0, 1],
            'url': urls[model_name],
            'std': stds[model_name],
            'mean': means[model_name],
            'num_classes': num_classes[dataset],
            'input_size': input_sizes[model_name],
        }


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.Conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):

    Conv3d = nn.Conv3d

    def __init__(self, block, layers, shortcut_type='B', num_classes=400):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = self.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # TAT
    #     self.tat_conv1 = self.Conv3d(64, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False) #28x28x16
    #     self.tat_conv2 = self.Conv3d(64, 21, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False) #14x14x16
    #     self.tat_conv3 = self.Conv3d(21, 1, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False) #7x7x16
    #    # self.tat_conv4 = self.Conv3d(256, 1, kernel_size=1, stride=(1, 1, 1), padding=(0, 0, 0), bias=False) #7x7x16x1
    #     self.tat_pooling = nn.MaxPool3d(kernel_size=(1, 7, 7))
    #     self.tat_norm = nn.Sigmoid()

        #self.ta = TABlock(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        #self.nl_2 = NONLocalBlock3D(in_channels=128)
        
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        #self.sta3 = STABlock(256,64)
        self.inplanes_tmp = self.inplanes
        self.layer4_v = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        #self.nl_4_v = NONLocalBlock3D(in_channels=512)
        self.inplanes = self.inplanes_tmp
        self.layer4_a = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        #self.nl_4_a = NONLocalBlock3D(in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        #self.fc_v = nn.Linear(512 * block.expansion, 1)
        #self.fc_a = nn.Linear(512 * block.expansion, 1)
        self.fc_v_cls = nn.Linear(512* block.expansion, 20)
        self.fc_a_cls = nn.Linear(512* block.expansion, 20)
        self.fc_expr = nn.Linear(512 * block.expansion * 2, 7)

        self.tanh = nn.Tanh()

        self.init_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    self.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, self.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_v = self.layer3_v(x)
        x_a = self.layer3_a(x)

        x_v = self.layer4_v(x_v)
        x_a = self.layer4_a(x_a)

        x_v = self.avgpool(x_v)
        x_a = self.avgpool(x_a)

        x_v = x_v.view(x_v.size(0), -1)
        x_v = self.fc_v(x_v)

        x_a = x_a.view(x_a.size(0), -1)
        x_a = self.fc_a(x_a)

        x_expr = self.fc_expr(torch.cat((x_v, x_a),1))
        x_expr = self.fc_expr(x_expr)

        #return self.tanh(x[:,0:2]), x[:,2:9]
        return torch.cat((x_v, x_a), 1), x_expr


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet3d10(**kwargs):
    """Constructs a ResNet3D-10 model."""
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    model = modify_resnets(model)
    return model


def resnet3d18(num_classes=400, pretrained='kinetics-400', shortcut_type='A', **kwargs):
    """Constructs a ResNet3D-18 model."""
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                     shortcut_type=shortcut_type, **kwargs)
    #pretrained = None
    if pretrained is not None:
        print("######## Pretrained #######")
        settings = pretrained_settings['resnet3d18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    else:
        print("######## NOT Pretrained #######")
    model = modify_resnets(model)
    return model


def resnet3d34(num_classes=400, pretrained='kinetics-400', shortcut_type='A', **kwargs):
    """Constructs a ResNet3D-34 model."""
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                     shortcut_type=shortcut_type, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d50(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-50 model."""
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d50'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d101(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-101 model."""
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d101'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d152(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-152 model."""
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet3d200(num_classes=400, pretrained='kinetics-400', **kwargs):
    """Constructs a ResNet3D-200 model."""
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet3d200'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resneti3d50(num_classes=400, pretrained='moments', **kwargs):
    """Constructs a ResNet3D-50 model."""
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained is not None:
        settings = torchvision_models.pretrained_settings['resnet50'][pretrained]
        model = inflate_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


if __name__ == '__main__':
    batch_size = 1
    num_frames = 48
    num_classes = 339
    img_feature_dim = 512
    frame_size = 224
    model = resnet3d50(num_classes=num_classes, pretrained='moments')

    input_var = torch.randn(batch_size, 3, num_frames, 224, 224)
    print(input_var.shape)
    output = model(input_var)
    print(output.shape)
