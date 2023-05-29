##from https://github.com/htdt/hyp_metric/blob/c89de0490691bacbd7332171c5455651fe49f25e/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import hyptorch.nn as hypnn


class VIT(nn.Module):
    def __init__(self, args):
        super(VIT, self).__init__()
        if args["model"].startswith("dino"):
            self.body = torch.hub.load("facebookresearch/dino:main", args["model"])
        else:
            print(f'getting {args["model"]} form TIMM')
            self.body = timm.create_model(args["model"], pretrained=True)

        if args["dataset"]=='CIFAR100':
            classes = 100
        elif args["dataset"]=='CalTech256':
            classes = 256
        elif args["dataset"]=='CUB':
            classes = 200

        self.n_classes = classes
        self.flattened_dim = 1000
        self.embedding_dim = 20

        self.fc1 = nn.Linear(self.flattened_dim, self.embedding_dim) ### mimic poincare ball
        self.fc2 = nn.Linear(self.embedding_dim, self.n_classes)


    def forward(self, x):
        x = self.body(x)
        # print('x.shape={}'.format(x.shape))
        e1 = F.relu(self.fc1(x))
        x = self.fc2(e1)
        return x, e1

    def get_embedding_dim(self):
        return self.embedding_dim



class HYPER_VIT(nn.Module):
    def __init__(self, args):
        super(HYPER_VIT, self).__init__()
        if args["model"].startswith("dino"):
            self.body = torch.hub.load("facebookresearch/dino:main", args["model"])
        else:
            self.body = timm.create_model(args["model"], pretrained=True)

        if args["dataset"]=='CIFAR100':
            classes = 100
        elif args["dataset"]=='CalTech256':
            classes = 256
        elif args["dataset"]=='CUB':
            classes = 200

        self.n_classes = classes
        self.flattened_dim = 1000

        self.fc1 = nn.Linear(self.flattened_dim, self.poincare_ball_dim)

        ## hyperparameters
        self.poincare_ball_dim = args['poincare_ball_dim'] #"Dimension of the Poincare ball"
        c = args['poincare_ball_curvature'] #1.0 #"Curvature of the Poincare ball"
        print(f'Using c={c} for HyperNet')
        train_x = False # train the exponential map origin
        train_c = False # train the Poincare ball curvature
        self.tp = hypnn.ToPoincare(
            c=c, train_x=train_x, train_c=train_c, ball_dim=self.poincare_ball_dim,
            riemannian=False,
            clip_r = 2.3,  # feature clipping radius
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.poincare_ball_dim, n_classes=self.n_classes, c=c)


    def forward(self, x):
        x = self.body(x)
        e1 = F.relu(self.fc1(x))
        e1_tp = self.tp(e1)
        return self.mlr(e1_tp, c=self.tp.c), e1_tp 


    def get_embedding_dim(self):
        return self.embedding_dim




def init_model_hyperVIT(cfg):
    cfg = cfg.as_dict()
    if cfg["model"].startswith("dino"):
        body = torch.hub.load("facebookresearch/dino:main", cfg["model"])
    else:
        body = timm.create_model(cfg["model"], pretrained=True)
    if cfg.get("hyp_c", 0) > 0:
        last = hypnn.ToPoincare(
            c=cfg["hyp_c"],
            ball_dim=cfg.get("emb", 128),
            riemannian=False,
            clip_r=cfg.get("clip_r", None),
        )
    else:
        last = NormLayer()
    bdim = 2048 if cfg["model"] == "resnet50" else 384
    head = nn.Sequential(nn.Linear(bdim, cfg.get("emb", 128)), last)
    nn.init.constant_(head[0].bias.data, 0)
    nn.init.orthogonal_(head[0].weight.data)
    rm_head(body)
    if cfg.get("freeze", None) is not None:
        freeze(body, cfg["freeze"])
    model = HeadSwitch(body, head)
    model.cuda().train()
    return model


class HeadSwitch(nn.Module):
    def __init__(self, body, head):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.head = head
        self.norm = NormLayer()

    def forward(self, x, skip_head=False):
        x = self.body(x)
        if type(x) == tuple:
            x = x[0]
        if not skip_head:
            x = self.head(x)
        else:
            x = self.norm(x)
        return x


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


def freeze(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])


def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())
