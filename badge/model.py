import torch.nn as nn
import torch.nn.functional as F
import hyptorch.nn as hypnn

def get_net(name):
    if name == 'MNIST':
        return Net1
    elif name == 'FashionMNIST':
        return Net1
    elif name == 'SVHN':
        return Net2
    elif name == 'CIFAR10':
        return Net3


class HyperNet(nn.Module):
    # https://github.com/leymir/hyperbolic-image-embeddings/blob/master/examples/mnist.py
    def __init__(self):
        ## hyperparameters
        self.poincare_ball_dim = 20 #"Dimension of the Poincare ball"
        c = 1.0 #"Curvature of the Poincare ball"
        train_x = False # train the exponential map origin
        train_c = False # train the Poincare ball curvature

        super(HyperNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, self.poincare_ball_dim)
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
        e1 = F.relu(self.fc1(x))
        e2 = self.fc2(e1)
        e2_tp = self.tp(e2)
        # return F.log_softmax(self.mlr(e2_tp, c=self.tp.c), dim=-1), e1. ##tmux 1 is running based on this
        return self.mlr(e2_tp, c=self.tp.c), e2_tp 

    def get_embedding_dim(self):
        # return 500 #if use after fc1 as embedding
        return self.poincare_ball_dim # if use after fc2 as embedding


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
        elif dataset=='CUB':
            classes = 200
            input_channel = 3
            self.flattened_dim = 50 * 13 * 13

        self.embedding_dim = 20

        # self.conv1 = nn.Conv2d(1, 20, 5, 1) #MNIST
        self.conv1 = nn.Conv2d(input_channel, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(self.flattened_dim, 500)
        self.fc2 = nn.Linear(500, self.embedding_dim) ### mimic poincare ball
        self.fc3 = nn.Linear(self.embedding_dim, classes)

    def forward(self, x):
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
