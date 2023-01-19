import torch
from torch import nn
from torchvision.models import swin_v2_t


class MyCustomSwinTiny(nn.Module):
    def __init__(self, input_channel, pretrained=False):
        super().__init__()
        if pretrained:
            from torchvision.models import Swin_V2_T_Weights
            swin_t = swin_v2_t(Swin_V2_T_Weights.IMAGENET1K_V1)
        else:
            swin_t = swin_v2_t()
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        self.features = nn.ModuleList(swin_t.children())[:-1]
        # Now we have our layers up to the fc layer, but we are not finished yet
        # we need to feed these to nn.Sequential() as well, this is needed because,
        # nn.ModuleList doesnt implement forward()
        # so you cant do sth like self.features(images). Therefore we use
        # nn.Sequential and since sequential doesnt accept lists, we
        # unpack all the items and send them like this
        if input_channel == 1:
            self.features[0][0][0] = nn.Conv2d(input_channel, 96, kernel_size=(4, 4), stride=(4, 4))

        self.features = nn.Sequential(*self.features)
        self.embDim = 768
        # now lets add our new layers
        # in_features = self.features.Flatten.in_features
        # from now, you can add any kind of layers in any quantity!
        # Here I'm creating two new layers
        self.classifier = nn.Linear(768, 10)
        # self.fc0_bn = nn.BatchNorm1d(256, eps=1e-2)
        # self.classifier = nn.Linear(256, 10)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, x):
        # now in forward pass, you have the full control,
        # we can use the feature part from our pretrained model  like this

        emb = self.features(x)
        # and also our new layers.
        # output = self.fc0_bn(F.relu(self.fc0(emb)))
        output = self.classifier(emb)

        return output, emb
    def get_embedding_dim(self):
        return self.embDim