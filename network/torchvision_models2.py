# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import re
import types
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

#################################################################
# You can find the definitions of those models here:
# https://github.com/pytorch/vision/blob/master/torchvision/models
#
# To fit the API, we usually added/redefined some methods and
# renamed some attributs (see below for each models).
#
# However, you usually do not need to see the original model
# definition from torchvision. Just use `print(model)` to see
# the modules and see bellow the `model.features` and
# `model.classifier` definitions.
#################################################################

__all__ = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'inceptionv3',
    'squeezenet1_0', 'squeezenet1_1',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'
]

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
    'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # 'vgg16_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth',
    # 'vgg19_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth'
}
moments_resnet50_url = ('http://moments.csail.mit.edu/moments_models/resnet50_moments-fd0c4436.pth')
places365_alexnet_url = 'http://pretorched-x.csail.mit.edu/models/alexnet_places365-0c3a7b83.pth'
places365_densenet161_url = 'http://pretorched-x.csail.mit.edu/models/densenet161_places365-62bbf0d4.pth'
places365_resnet_urls = {
    'resnet18': 'http://pretorched-x.csail.mit.edu/models/resnet18_places365-dbad67aa.pth',
    'resnet50': 'http://pretorched-x.csail.mit.edu/models/resnet50_places365-a570fcfc.pth'}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ['inceptionv3']:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]

pretrained_settings = {}

for model_name in __all__:

    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 1000
        }
    }

# Add Moments pretrained model.
pretrained_settings['resnet50'].update({
    'moments': {
        'url': moments_resnet50_url,
        'input_space': 'RGB',
        'input_size': input_sizes['resnet50'],
        'input_range': [0, 1],
        'mean': means['resnet50'],
        'std': stds['resnet50'],
        'num_classes': 339,
    }
})

pretrained_settings['alexnet'].update({
    'places365': {
        'url': places365_alexnet_url,
        'input_space': 'RGB',
        'input_size': input_sizes['alexnet'],
        'input_range': [0, 1],
        'mean': means['alexnet'],
        'std': stds['alexnet'],
        'num_classes': 365,
    }
})

pretrained_settings['densenet161'].update({
    'places365': {
        'url': places365_densenet161_url,
        'input_space': 'RGB',
        'input_size': input_sizes['densenet161'],
        'input_range': [0, 1],
        'mean': means['densenet161'],
        'std': stds['densenet161'],
        'num_classes': 365,
    }
})
# Add Places 365 pretrained model.
for model_name, url in places365_resnet_urls.items():
    pretrained_settings[model_name].update({
        'places365': {
            'url': url,
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 365,
        }
    })


# for model_name in ['vgg16', 'vgg19']:
#     pretrained_settings[model_name]['imagenet_caffe'] = {
#         'url': model_urls[model_name + '_caffe'],
#         'input_space': 'BGR',
#         'input_size': input_sizes[model_name],
#         'input_range': [0, 255],
#         'mean': [103.939, 116.779, 123.68],
#         'std': [1., 1., 1.],
#         'num_classes': 1000
#     }


def load_pretrained(model, num_classes, settings):
    
    model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model


def inflate_pretrained(model, num_classes, settings):

    def inflate(shape, w, dim=2):
        return w.unsqueeze(dim).expand(shape)

    assert num_classes == settings['num_classes'], \
        "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    ws = model.state_dict()
    weights = model_zoo.load_url(settings['url'])
    for x in weights:
        for y in ws:
            if x == y:
                if weights[x].shape != ws[y].shape:
                    n = weights[x].unsqueeze(2).expand_as(ws[y])
                    weights[x] = n
    model.load_state_dict(weights)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model


#################################################################
# AlexNet


def modify_alexnet(model):
    # Modify attributs
    model._features = model.features
    del model.features
    model.dropout0 = model.classifier[0]
    model.linear0 = model.classifier[1]
    model.relu0 = model.classifier[2]
    model.dropout1 = model.classifier[3]
    model.linear1 = model.classifier[4]
    model.relu1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    setattr(model.__class__, 'features', features)
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    return model


def alexnet(num_classes=1000, pretrained='imagenet'):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    model = models.alexnet(num_classes=num_classes, pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['alexnet'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_alexnet(model)
    return model

###############################################################
# DenseNets


def modify_densenets(model):
    # Modify attributs
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    return model


def densenet121(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet121(num_classes=num_classes, pretrained=False)
    if pretrained is not None:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        settings = pretrained_settings['densenet121'][pretrained]
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(settings['url'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet169(num_classes=num_classes, pretrained=False)
    if pretrained is not None:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        settings = pretrained_settings['densenet169'][pretrained]
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(settings['url'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet201(num_classes=num_classes, pretrained=False)
    if pretrained is not None:
       # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        settings = pretrained_settings['densenet201'][pretrained]
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(settings['url'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet161(num_classes=num_classes, pretrained=False)
    if pretrained is not None:
       # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        settings = pretrained_settings['densenet161'][pretrained]
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(settings['url'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    model = modify_densenets(model)
    return model

###############################################################
# InceptionV3


def inceptionv3(num_classes=1000, pretrained='imagenet'):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    """
    model = models.inception_v3(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['inceptionv3'][pretrained]
        model = load_pretrained(model, num_classes, settings)

    # Modify attributs
    model.last_linear = model.fc
    del model.fc

    def features(self, input):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(input)  # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)  # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)  # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 35 x 35 x 192
        x = self.Mixed_5b(x)  # 35 x 35 x 256
        x = self.Mixed_5c(x)  # 35 x 35 x 288
        x = self.Mixed_5d(x)  # 35 x 35 x 288
        x = self.Mixed_6a(x)  # 17 x 17 x 768
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768
        if self.training and self.aux_logits:
            self._out_aux = self.AuxLogits(x)  # 17 x 17 x 768
        x = self.Mixed_7a(x)  # 8 x 8 x 1280
        x = self.Mixed_7b(x)  # 8 x 8 x 2048
        x = self.Mixed_7c(x)  # 8 x 8 x 2048
        return x

    def logits(self, features):
        x = F.avg_pool2d(features, kernel_size=8)  # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)  # 1 x 1 x 2048
        x = x.view(x.size(0), -1)  # 2048
        x = self.last_linear(x)  # 1000 (num_classes)
        if self.training and self.aux_logits:
            aux = self._out_aux
            self._out_aux = None
            return x, aux
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

###############################################################
# ResNets


def modify_resnets(model):
    # Modify attributs
    # 修改output layer
    model.last_linear = nn.Linear(512 * 1, 9)
    # 初始化
    nn.init.xavier_normal_(model.last_linear.weight)
    model.last_linear.bias.data.fill_(0.01)

    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        debug = False
        if debug:
            print(x.shape)
        x = self.layer1(x)
        if debug:
            print(x.shape)
        x = self.layer2(x)
        if debug:
            print(x.shape)
        x = self.layer3(x)
        if debug:
            print(x.shape)

        x_v = self.layer4_v(x)
        #x_v = self.nl_4_v(x_v)
        x_a = self.layer4_a(x)
        if debug:
            print(x_v.shape)
        #x_a = self.nl_4_a(x_a)

        return (x_v, x_a)

    def logits(self, features):

        x_v = features[0] # [b, 512, 4, 4, 4]
        x_a = features[1]
        
        #x_v = self.avgpool(x_v) # [b, 512, 4, 4, 4]
        #x_a = self.avgpool(x_a)
        batch_size = x_v.shape[0]
        x1_v = self.sp_pool(x_v[:,:,0,:,:]).view(batch_size,-1)
        x2_v = self.sp_pool(x_v[:,:,1,:,:]).view(batch_size,-1)
        x3_v = self.sp_pool(x_v[:,:,2,:,:]).view(batch_size,-1)
        x4_v = self.sp_pool(x_v[:,:,3,:,:]).view(batch_size,-1)

        x1_a = self.sp_pool(x_a[:,:,0,:,:]).view(batch_size,-1)
        x2_a = self.sp_pool(x_a[:,:,1,:,:]).view(batch_size,-1)
        x3_a = self.sp_pool(x_a[:,:,2,:,:]).view(batch_size,-1)
        x4_a = self.sp_pool(x_a[:,:,3,:,:]).view(batch_size,-1)
        

        #x_v = x_v.view(x_v.size(0), -1)
        #x_a = x_a.view(x_a.size(0), -1)


        x_vs = x1_v + x2_v + x3_v + x4_v
        x_as = x1_a + x2_a + x3_a + x4_a

        x1_vc = torch.cat((x1_v, x_vs),1)
        x2_vc = torch.cat((x2_v, x_vs),1)
        x3_vc = torch.cat((x3_v, x_vs),1)
        x4_vc = torch.cat((x4_v, x_vs),1)

        x1_ac = torch.cat((x1_a, x_as),1)
        x2_ac = torch.cat((x2_a, x_as),1)
        x3_ac = torch.cat((x3_a, x_as),1)
        x4_ac = torch.cat((x4_a, x_as),1)

        beta_v1 = self.sigmoid(self.beta(self.dropout(x1_vc)))
        beta_v2 = self.sigmoid(self.beta(self.dropout(x2_vc)))
        beta_v3 = self.sigmoid(self.beta(self.dropout(x3_vc)))
        beta_v4 = self.sigmoid(self.beta(self.dropout(x4_vc)))
        beta_a1 = self.sigmoid(self.beta(self.dropout(x1_ac)))
        beta_a2 = self.sigmoid(self.beta(self.dropout(x2_ac)))
        beta_a3 = self.sigmoid(self.beta(self.dropout(x3_ac)))
        beta_a4 = self.sigmoid(self.beta(self.dropout(x4_ac)))
        sum_beta_v =  beta_v1 + beta_v2 + beta_v3 + beta_v4
        sum_beta_a = beta_a1 + beta_a2 + beta_a3 + beta_a4


        x_expr = torch.cat((x1_v*beta_v1/sum_beta_v, x2_v*beta_v2/sum_beta_v, 
                            x3_v*beta_v3/sum_beta_v, x4_v*beta_v4/sum_beta_v, 
                            x1_a*beta_a1/sum_beta_a, x2_a*beta_a2/sum_beta_a, 
                            x3_a*beta_a3/sum_beta_a, x4_a*beta_a4/sum_beta_a),1)
        x_v = torch.cat((x1_v*beta_v1/sum_beta_v, x2_v*beta_v2/sum_beta_v, 
                            x3_v*beta_v3/sum_beta_v, x4_v*beta_v4/sum_beta_v),1)
        x_a = torch.cat((x1_v*beta_v1/sum_beta_v, x2_v*beta_v2/sum_beta_v, 
                            x3_v*beta_v3/sum_beta_v, x4_v*beta_v4/sum_beta_v),1)

        x_v_cls = self.fc_v_cls(x_v)
        x_a_cls = self.fc_a_cls(x_a)
        
        x_expr = self.fc_expr(x_expr)
        return torch.cat((x_v_cls, x_a_cls), 1), x_expr
        

    def forward(self, input):
        # x = self.features(input)
        # x = self.logits(x)
        # return nn.Tanh()(x[:,0:2]), x[:,2:9]

        x = self.features(input)
        x_va, x_expr = self.logits(x)
        return (x_va), x_expr

    # Modify methods
    setattr(model.__class__, 'features', features)
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    # model.features = types.MethodType(features, model)
    # model.logits = types.MethodType(logits, model)
    # model.forward = types.MethodType(forward, model)
    # model.features = types.MethodType(features, model)
    # model.logits = types.MethodType(logits, model)
    # model.forward = types.MethodType(forward, model)
    return model


def resnet18(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        print("######## Pretrained #######")
        settings = pretrained_settings['resnet18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    else:
        print("######## NOT Pretrained #######")
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet50'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-101 model.
    """
    model = models.resnet101(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet101'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model

###############################################################
# SqueezeNets


def modify_squeezenets(model):
    # /!\ Beware squeezenets do not have any last_linear module

    # Modify attributs
    model.dropout = model.classifier[0]
    model.last_conv = model.classifier[1]
    model.relu = model.classifier[2]
    model.avgpool = model.classifier[3]
    del model.classifier

    def logits(self, features):
        x = self.dropout(features)
        x = self.last_conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def squeezenet1_0(num_classes=1000, pretrained='imagenet'):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = models.squeezenet1_0(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['squeezenet1_0'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained='imagenet'):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = models.squeezenet1_1(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['squeezenet1_1'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model

###############################################################
# VGGs


def modify_vggs(model):
    # Modify attributs
    model._features = model.features
    del model.features
    model.linear0 = model.classifier[0]
    model.relu0 = model.classifier[1]
    model.dropout0 = model.classifier[2]
    model.linear1 = model.classifier[3]
    model.relu1 = model.classifier[4]
    model.dropout1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), -1)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout0(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.dropout1(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    setattr(model.__class__, 'features', features)
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    return model


def vgg11(num_classes=1000, pretrained='imagenet'):
    """VGG 11-layer model (configuration "A")
    """
    model = models.vgg11(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg11'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    model = models.vgg11_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg11_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained='imagenet'):
    """VGG 13-layer model (configuration "B")
    """
    model = models.vgg13(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg13'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    model = models.vgg13_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg13_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained='imagenet'):
    """VGG 16-layer model (configuration "D")
    """
    model = models.vgg16(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg16'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = models.vgg16_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg16_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained='imagenet'):
    """VGG 19-layer model (configuration "E")
    """
    model = models.vgg19(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg19'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    model = models.vgg19_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg19_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model
