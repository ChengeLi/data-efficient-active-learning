import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import hyptorch.nn as hypnn
import hyptorch.pmath as pmath
from resnet import ResNet50

import pdb

def get_net(name):
    if name == 'MNIST':
        return Net1
    elif name == 'FashionMNIST':
        return Net1
    elif name == 'SVHN':
        return Net2
    elif name == 'CIFAR10':
        return Net3
    # elif name == 'CUB':
    #     return ResNet50


# linear model class
class linMod(nn.Module):
    def __init__(self, opts, dim=28):
        super(linMod, self).__init__()
        self.dim = dim
        self.lm = nn.Linear(dim, opts.nClasses)

    def forward(self, x):
        x = x.view(-1, self.dim)
        out = self.lm(x)
        return out, x

    def get_embedding_dim(self):
        return self.dim


# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, opts, embSize=128, useNonLin=True):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, embSize)
        self.linear = nn.Linear(embSize, opts.nClasses, bias=False)
        self.useNonLin = useNonLin

    def forward(self, x):
        x = x.view(-1, self.dim)
        if self.useNonLin:
            emb = F.relu(self.lm1(x))
        else:
            emb = self.lm1(x)
        out = self.linear(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize


class HyperNet(nn.Module):
    # https://github.com/leymir/hyperbolic-image-embeddings/blob/master/examples/mnist.py
    def __init__(self, args):
        ## hyperparameters
        self.poincare_ball_dim = args['poincare_ball_dim'] #"Dimension of the Poincare ball"
        c = args['poincare_ball_curvature'] #1.0 #"Curvature of the Poincare ball"
        print(f'Using c={c} for HyperNet')
        train_x = False # train the exponential map origin
        train_c = False # train the Poincare ball curvature

        super(HyperNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)  #cifar10 original image size is 32x32
        self.fc2 = nn.Linear(500, self.poincare_ball_dim)
        self.tp = hypnn.ToPoincare(
            c=c, train_x=train_x, train_c=train_c, ball_dim=self.poincare_ball_dim,
            riemannian=False,
            clip_r = 2.3,  # feature clipping radius
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.poincare_ball_dim, n_classes=10, c=c)
        # self.mlr = hypnn.HyperbolicMLR_fix_grad(ball_dim=dim, n_classes=10, c=c)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4 * 4 * 50)
        x = x.view(-1, 5 * 5 * 50)
        e1 = F.relu(self.fc1(x))
        e2 = self.fc2(e1)
        e2_tp = self.tp(e2)
        return self.mlr(e2_tp, c=self.tp.c), e2_tp 
        # return self.mlr(e2_tp, c=self.tp.c), e1 

    def get_embedding_dim(self):
        # return 500 #if use after fc1 as embedding
        return self.poincare_ball_dim # if use after fc2 as embedding


class HyperNet2(nn.Module):
    def __init__(self, args):
        ## hyperparameters
        self.poincare_ball_dim = args['poincare_ball_dim']
        self.c = args['poincare_ball_curvature']
        print(f'Using c={self.c} for HyperNet')
        train_x = False # train the exponential map origin
        train_c = False # train the Poincare ball curvature

        super(HyperNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.hyfc1 = hypnn.HypLinear(4 * 4 * 50, 500, c=self.c)

        self.hyfc1 = hypnn.HypLinear(5 * 5 * 50, 500, c=self.c)
        self.hyfc2 = hypnn.HypLinear(500, self.poincare_ball_dim, c=self.c)
        self.tp = hypnn.ToPoincare(
            c=self.c, train_x=train_x, train_c=train_c, ball_dim=self.poincare_ball_dim,
            clip_r = 2.3,  # feature clipping radius
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.poincare_ball_dim, n_classes=10, c=self.c)
        # self.mlr = hypnn.HyperbolicMLR_fix_grad(ball_dim=dim, n_classes=10, c=self.c)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4 * 4 * 50)
        x = x.view(-1, 5 * 5 * 50)
        e1 = F.relu(self.hyfc1(x))
        e2 = self.hyfc2(e1)
        e2_tp = self.tp(e2)
        # return F.log_softmax(self.mlr(e2_tp, c=self.tp.c), dim=-1), e1
        return self.mlr(e2_tp, c=self.tp.c), e2_tp 
        
        # logits1 = self.mlr(e2_tp, c=self.tp.c)
        # proto = pmath.poincare_mean(e2_tp, dim=0, c=self.c)
        # temperature = 1        
        # # e2_tp = e2_tp.view(-1, )
        # logits2 = (
        #         -pmath.dist_matrix(e2_tp, proto, c=self.c) / temperature
        #     )
        # print(logits1, logits2)
        # pdb.set_trace()


    def get_embedding_dim(self):
        return self.poincare_ball_dim # if use after fc2 as embedding



class HyperNet3(nn.Module):
    def __init__(self, args):
        ## hyperparameters
        self.poincare_ball_dim = args['poincare_ball_dim'] #"Dimension of the Poincare ball"
        c = args['poincare_ball_curvature'] #1.0 #"Curvature of the Poincare ball"
        print(f'Using c={c} for HyperNet')
        train_x = False # train the exponential map origin
        train_c = False # train the Poincare ball curvature

        super(HyperNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc1 = nn.Linear(5 * 5 * 50, 500)  #cifar10 original image size is 32x32
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, self.poincare_ball_dim)
        self.tp = hypnn.ToPoincare(
            c=c, train_x=train_x, train_c=train_c, ball_dim=self.poincare_ball_dim
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.poincare_ball_dim, n_classes=10, c=c)
        # self.mlr = hypnn.HyperbolicMLR_fix_grad(ball_dim=dim, n_classes=10, c=c)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        # x = x.view(-1, 5 * 5 * 50)
        e1 = F.relu(self.fc1(x))
        e2 = F.relu(self.fc2(e1))
        e3 = self.fc3(e2)
        e3_tp = self.tp(e3)
        return self.mlr(e3_tp, c=self.tp.c), e3_tp 

    def get_embedding_dim(self):
        return self.poincare_ball_dim # if use after fc2 as embedding


class HyperResNet50(nn.Module):
    def __init__(self, args):
        super(HyperResNet50, self).__init__()
        ## hyperparameters
        self.poincare_ball_dim = args['poincare_ball_dim'] #"Dimension of the Poincare ball"
        c = args['poincare_ball_curvature'] #1.0 #"Curvature of the Poincare ball"
        print(f'Using c={c} for HyperResNet50')
        train_x = False # train the exponential map origin
        train_c = False # train the Poincare ball curvature
        
        self.n_classes = 200 #CUB
        self.resnet50 = ResNet50(dataset='CUB',num_classes=self.n_classes)
        self.resnet_out_dim = 512 #2048 #512*3*3 
        self.fc1 = nn.Linear(self.resnet_out_dim, self.poincare_ball_dim)
        self.tp = hypnn.ToPoincare(
            c=c, train_x=train_x, train_c=train_c, ball_dim=self.poincare_ball_dim,
            clip_r = 2.3,  # feature clipping radius
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.poincare_ball_dim, n_classes=self.n_classes, c=c)
        # self.mlr = hypnn.HyperbolicMLR_fix_grad(ball_dim=dim, n_classes=self.n_classes, c=c)

    def forward(self, x):
        _, x = self.resnet50(x)
        x = x.view(-1, self.resnet_out_dim)
        e1 = F.relu(self.fc1(x))
        e1_tp = self.tp(e1)
        return self.mlr(e1_tp, c=self.tp.c), e1 

    def get_embedding_dim(self):
        return self.poincare_ball_dim




class Net0(nn.Module):
    ### This is hyperNet without the last layer
    def __init__(self):
        super(Net0, self).__init__()
        dim = 10
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 500


class Net00(nn.Module):
    ### This is hyperNet without the last layer
    def __init__(self, dataset):
        super(Net00, self).__init__()
        if dataset=='MNIST':
            classes = 10
            input_channel = 1
            self.flattened_dim = 4 * 4 * 50
        elif dataset=='CIFAR10':
            classes = 10
            input_channel = 3
            self.flattened_dim = 5 * 5 * 50
        elif dataset=='CIFAR100':
            classes = 100
            input_channel = 3
            self.flattened_dim = 5 * 5 * 50
        elif dataset=='CalTech256':
            classes = 256
            input_channel = 3
            self.flattened_dim = 6 * 6 * 50
        elif dataset=='CUB':
            classes = 200
            input_channel = 3
            self.flattened_dim = 50 * 13 * 13

        self.embedding_dim = 20
        self.dataset = dataset
        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #MNIST
        if self.dataset=='CalTech256':
            self.conv1 = nn.Conv2d(input_channel, 20, 5, 2)
            self.conv2 = nn.Conv2d(20, 30, 5, 2)
            self.conv3 = nn.Conv2d(30, 50, 3, 2)
        else:
            self.conv1 = nn.Conv2d(input_channel, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(self.flattened_dim, 500)
        self.fc2 = nn.Linear(500, self.embedding_dim) ### mimic poincare ball
        self.fc3 = nn.Linear(self.embedding_dim, classes)

    def forward(self, x, embedding=False):
        if embedding:
            e2 = x
        else:
            # print('0')
            # print(x.size())
            x = F.relu(self.conv1(x))
            # print('1')
            # print(x.size())
            x = F.max_pool2d(x, 2, 2)
            # print('2')
            # print(x.size())
            x = F.relu(self.conv2(x))
            # print('3')
            # print(x.size())
            if self.dataset == 'CalTech256':
                x = F.relu(self.conv3(x))
            # print('44')
            # print(x.size())
            x = F.max_pool2d(x, 2, 2)
            # print('4')
            # print(x.size())
            x = x.view(-1, self.flattened_dim)
            # print('5')
            # print(x.size())
            e1 = F.relu(self.fc1(x))
            # print('6')
            # print(x.size())
            x = F.dropout(e1, training=self.training)
            # print('7')
            # print(x.size())
            e2 = F.relu(self.fc2(x))
            # print('8')
            # print(x.size())
        x = F.dropout(e2, training=self.training)
        # print('9')
        # print(x.size())
        x = self.fc3(x)
        # print('10')
        # print(x.size())
        return x, e2

    def get_embedding_dim(self):
        return self.embedding_dim


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
