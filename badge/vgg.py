'''VGG11/13/16/19 in Pytorch.'''
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import hyptorch.nn as hypnn
import hyptorch.pmath as pmath

# ## From badge implementation
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_from_VAAL(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG_from_VAAL, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes)
        )
        self.final_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        # print(x.size())
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # emb = out.view(out.size(0), -1)
        emb = self.classifier(out)
        out = self.final_layer(emb)

        return out, emb

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 512

class VGG(nn.Module):
    def __init__(self, vgg_name, dataset, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.dataset = dataset
        if dataset=='CUB' or dataset=='CalTech256':
            self.linear1 = nn.Linear(25088, 1024)
            self.linear2 = nn.Linear(1024, 512)
            self.classifier = nn.Linear(512, num_classes)
        elif dataset=='CIFAR100' or dataset=='CIFAR10':
            self.classifier = nn.Linear(512, num_classes)


    def forward(self, x, embedding=False):
        if embedding:
            emb = x
        else:
            out = self.features(x)
            # print(x.size())
            # print(out.size())
            if self.dataset=='CUB' or self.dataset=='CalTech256':
                out = F.relu(self.linear1(out.view(out.size(0), -1)))
                emb = F.relu(self.linear2(out))
            elif self.dataset=='CIFAR100' or self.dataset=='CIFAR10':
                emb = out.view(out.size(0), -1)
            else:
                sys.exit('dataset not define in the model')
        out = self.classifier(emb)
        return out, emb

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 512

# def test():
#     net = VGG('VGG11')
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())

## From VAAL implementation
# import torch
# import torch.nn as nn
#
#
# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
#
# from torch.hub import load_state_dict_from_url
#
# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }
#
#
# class VGG(nn.Module):
#
#     def __init__(self, features, num_classes=1000, init_weights=True):
#         super(VGG, self).__init__()
#         self.features = features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             # nn.Linear(4096, num_classes)
#         )
#         self.final_layer = nn.Linear(4096, num_classes)
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         out = self.features(x)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         emb = self.classifier(out)
#         # emb = torch.flatten(out, 1)
#         # out = self.classifier(out)
#         return out, emb
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def get_embedding_dim(self):
#         return 4096
#
# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#

# cfgs = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
#
# def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def vgg11(pretrained=False, progress=True, **kwargs):
#     r"""VGG 11-layer model (configuration "A") from
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)
#
#
# def vgg11_bn(pretrained=False, progress=True, **kwargs):
#     r"""VGG 11-layer model (configuration "A") with batch normalization
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)
#
#
# def vgg13(pretrained=False, progress=True, **kwargs):
#     r"""VGG 13-layer model (configuration "B")
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)
#
#
# def vgg13_bn(pretrained=False, progress=True, **kwargs):
#     r"""VGG 13-layer model (configuration "B") with batch normalization
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)
#
#
# def vgg16(pretrained=False, progress=True, **kwargs):
#     r"""VGG 16-layer model (configuration "D")
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
#
#
# def vgg16_bn(pretrained=False, progress=True, **kwargs):
#     r"""VGG 16-layer model (configuration "D") with batch normalization
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)
#
#
# def vgg19(pretrained=False, progress=True, **kwargs):
#     r"""VGG 19-layer model (configuration "E")
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)
#
#
# def vgg19_bn(pretrained=False, progress=True, **kwargs):
#     r"""VGG 19-layer model (configuration 'E') with batch normalization
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


class HyperVGG(nn.Module):
    def __init__(self, vgg_name, dataset, num_classes, args):
        super(HyperVGG, self).__init__()
        ## hyperparameters
        self.poincare_ball_dim = args['poincare_ball_dim'] #"Dimension of the Poincare ball"
        c = args['poincare_ball_curvature'] #1.0 #"Curvature of the Poincare ball"
        print(f'Using c={c} for HyperNet')
        train_x = False # train the exponential map origin
        train_c = False # train the Poincare ball curvature


        self.features = self._make_layers(cfg[vgg_name])
        self.dataset = dataset
        if dataset=='CUB' or dataset=='CalTech256':
            self.linear1 = nn.Linear(25088, 1024)
            self.linear2 = nn.Linear(1024, 512)
            self.classifier = nn.Linear(512, self.poincare_ball_dim)
        elif dataset=='CIFAR100' or dataset=='CIFAR10':
            self.classifier = nn.Linear(512, self.poincare_ball_dim)

        self.tp = hypnn.ToPoincare(
            c=c, train_x=train_x, train_c=train_c, ball_dim=self.poincare_ball_dim,
            riemannian=False,
            clip_r = 2.3,
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.poincare_ball_dim, n_classes=num_classes, c=c)


    def forward(self, x, embedding=False):
        if embedding:
            e1 = x
        else:
            out = self.features(x)
            # print(x.size())
            # print(out.size())
            if self.dataset=='CUB' or self.dataset=='CalTech256':
                out = F.relu(self.linear1(out.view(out.size(0), -1)))
                emb = F.relu(self.linear2(out))
            elif self.dataset=='CIFAR100' or self.dataset=='CIFAR10':
                emb = out.view(out.size(0), -1)
            else:
                sys.exit('dataset not define in the model')
            e1 = self.classifier(emb)

        e2_tp = self.tp(e1)
        return self.mlr(e2_tp, c=self.tp.c), e2_tp 


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return self.poincare_ball_dim # if use after fc2 as embedding
