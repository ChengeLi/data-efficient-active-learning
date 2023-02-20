import numpy as np
import sys
import gc
import gzip
import pickle
# import openml
import os
import argparse
from tqdm import tqdm
# from torch.nn import Linear, Sequential

import swin
from query_strategies.util import create_directory
from query_strategies.hyperbolic_embedding_umap_sampling import HypUmapSampling, HypNetBadgeSampling, \
    UmapPoincareKmeansSampling, UmapHyperboloidKmeansSampling, UmapHyperboloidKmeansSampling2, \
    HyperboloidKmeansSampling, PoincareKmeansSampling, HypNetNormSampling, UmapKmeansSampling, BadgePoincareSampling
from dataset import get_dataset, get_handler
# from model import get_net
from model import HyperNet, Net0, Net00
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import random
# import time
import pdb
# from scipy.stats import zscore
from query_strategies import RandomSampling, BadgeSampling, \
    BaselineSampling, LeastConfidence, MarginSampling, \
    EntropySampling, ActiveLearningByLearning, BaitSampling, CoreSet
# , , ActiveLearningByLearning, \
# LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
# KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
# AdversarialBIM, AdversarialDeepFool,

parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
parser.add_argument('--nEnd', help='total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=128)
parser.add_argument('--rounds', help='number of rounds (0 does entire dataset)', type=int, default=0)
parser.add_argument('--trunc', help='dataset truncation (-1 is no truncation)', type=int, default=-1)
parser.add_argument('--aug', help='do augmentation (for cifar)', type=int, default=0)
parser.add_argument('--dummy', help='dummy input for indexing replicates', type=int, default=1)
opts = parser.parse_args()
print(opts, flush=True)

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)
# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB) / opts.nQuery)
DATA_NAME = opts.data
# regularization settings for bait
opts.lamb = 1
visualize_embedding = True
visualize_learningcurve = True
# non-openml data defaults
args_pool = {'MNIST':
                 {'n_epoch': 10,
                  'max_epoch': 100,
                  'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                  'loader_tr_args': {'batch_size': 256, 'num_workers': 0},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'FashionMNIST':
                 {'n_epoch': 10,
                  'max_epoch': 100,
                  'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                  'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'SVHN':
                 {'n_epoch': 20,
                  'max_epoch': 100,
                  'transform': transforms.Compose(
                     [transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                  'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'CIFAR10':
                 {'n_epoch': 3,
                  'max_epoch': 100,
                  'transform': transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                 ]),
                  'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
                  'transformTest': transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2470, 0.2435, 0.2616))])}
             }
opts.nClasses = 10
if opts.model == 'swin_t':
    args_pool['max_epoch'] = 200
    args_pool['CIFAR10']['transform'] = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # should I change this to the values for Swin_V2_t? (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transforms.Resize((256, 256))])

if opts.aug == 0:
    args_pool['CIFAR10']['transform'] = args_pool['CIFAR10']['transformTest']  # remove data augmentation
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['FashionMNIST']['transformTest'] = args_pool['FashionMNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']

if opts.did == 0: args = args_pool[DATA_NAME]
if not os.path.exists(opts.path):
    os.makedirs(opts.path)

# load openml dataset if did is supplied
if opts.did > 0:
    data = pickle.load(open('oml/data_' + str(opts.did) + '.pk', 'rb'))['data']
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)
    opts.nClasses = int(max(y) + 1)
    nSamps, opts.dim = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split = int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == opts.nClasses: break

    args = {'transform': transforms.Compose([transforms.ToTensor()]),
            'n_epoch': 10,
            'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            'optimizer_args': {'lr': 0.01, 'momentum': 0},
            'transformTest': transforms.Compose([transforms.ToTensor()])}
    handler = get_handler('other')

# load non-openml dataset
else:
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
    opts.dim = np.shape(X_tr)[1:]
    handler = get_handler(opts.data)

if opts.trunc != -1:
    inds = np.random.permutation(len(X_tr))[:opts.trunc]
    X_tr = X_tr[inds]
    Y_tr = Y_tr[inds]
    inds = torch.where(Y_tr < 10)[0]
    X_tr = X_tr[inds]
    Y_tr = Y_tr[inds]
    opts.nClasses = int(max(Y_tr) + 1)

args['lr'] = opts.lr
args['modelType'] = opts.model
args['lamb'] = opts.lamb
if 'CIFAR' in opts.data: args['lamb'] = 1e-2

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)
print('number of rounds: {}'.format(NUM_ROUND), flush=True)
print('Training batch size: {}'.format(args_pool[opts.data]['loader_tr_args']['batch_size']), flush=True)
print('Testing batch size: {}'.format(args_pool[opts.data]['loader_te_args']['batch_size']), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True


# linear model class
class linMod(nn.Module):
    def __init__(self, dim=28):
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
    def __init__(self, dim, embSize=128, useNonLin=True):
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


EXPERIMENT_NAME = DATA_NAME + '_' + opts.model + '_' + opts.alg + '_' + str(NUM_QUERY)
args['output_dir'] = os.path.join('./badge/output', EXPERIMENT_NAME)
create_directory(args['output_dir'])
# load specified network
if opts.model == 'mlp':
    net = mlpMod(opts.dim, embSize=opts.nEmb)
elif opts.model == 'resnet':
    net = resnet.ResNet18()
elif opts.model == 'vgg':
    net = vgg.VGG('VGG16')
elif opts.model == 'lin':
    dim = np.prod(list(X_tr.shape[1:]))
    net = linMod(dim=int(dim))
elif opts.model == 'swin_t':
    if opts.data == 'MNIST':
        net = swin.MyCustomSwinTiny(input_channel=1)
    else:
        net = swin.MyCustomSwinTiny(input_channel=3, pretrained=True) #, pretrained=True
elif opts.model == 'HyperNet':
    print('Using hypernet')
    net = HyperNet()
elif opts.model == 'net0':
    print('Using Net0')
    net = Net0()
elif opts.model == 'net00':
    print('Using Net00')
    net = Net00()
else:
    print('choose a valid model - mlp, resnet, or vgg', flush=True)
    raise ValueError

if opts.did > 0 and opts.model != 'mlp':
    print('openML datasets only work with mlp', flush=True)
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

# set up the specified sampler
if opts.alg == 'rand':  # random sampling
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'bait':  # bait sampling
    strategy = BaitSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'conf':  # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'marg':  # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'badge':  # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'hypUmap':
    strategy = HypUmapSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'umap':
    strategy = UmapKmeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'PoincareKmeans':
    strategy = PoincareKmeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'BadgePoincareKmeans':
    strategy = BadgePoincareSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'HyperboloidKmeans':
    strategy = HyperboloidKmeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'UmapPoincareKmeans':
    strategy = UmapPoincareKmeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'UmapHyperboloidKmeans':
    strategy = UmapHyperboloidKmeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'UmapHyperboloidKmeans2':
    strategy = UmapHyperboloidKmeansSampling2(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'hypNetBadge':
    strategy = HypNetBadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'hypNetNorm':
    strategy = HypNetNormSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'coreset':  # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'entropy':  # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'baseline':  # badge but with k-DPP sampling instead of k-means++
    strategy = BaselineSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'albl':  # active learning by learning
    albl_list = [LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args),
                 CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)]
    strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
else:
    print('choose a valid acquisition function', flush=True)
    raise ValueError

# print info
if opts.did > 0: DATA_NAME = 'OML' + str(opts.did)
print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)

if type(X_te) == torch.Tensor: X_te = X_te.numpy()
results = []
# round 0 accuracy
strategy.train(verbose=False,model_selection=opts.model)
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND + 1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print(str(opts.nStart) + '\ttesting accuracy {}'.format(acc[0]), flush=True)
results.append([sum(idxs_lb), acc[0]])


for rd in tqdm(range(1, NUM_ROUND + 1)):
    print('')
    print('Round {}'.format(rd), flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # query
    output = strategy.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train(verbose=False, model_selection=opts.model)

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]), flush=True)
    results.append([sum(idxs_lb), acc[rd]])
    if sum(~strategy.idxs_lb) < opts.nQuery: break
    if opts.rounds > 0 and rd == (opts.rounds - 1): break

results = np.asarray(results)
np.savetxt(os.path.join(args['output_dir'],EXPERIMENT_NAME+'_strategy_performance.txt'), results)

if visualize_embedding:
    import cv2
    import numpy as np

    images_dir = os.path.join(args['output_dir'],'images')
    if not os.path.exists(args['output_dir']):
        create_directory(args['output_dir'])
    if not os.path.exists(images_dir):
        create_directory(images_dir)
    if len(os.listdir(images_dir))>0:
        img_array = []
        for filename in sorted(os.listdir(images_dir)):
            img = cv2.imread(os.path.join(images_dir, filename))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(os.path.join(args['output_dir'],EXPERIMENT_NAME+'_demo_emb.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 3, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

if visualize_learningcurve:
    import matplotlib.pyplot as plt
    results = np.loadtxt(os.path.join(args['output_dir'],EXPERIMENT_NAME+'_strategy_performance.txt'))
    fig = plt.figure()
    plt.plot(results[:,0], results[:,1], label=opts.alg)
    plt.scatter(results[:,0], results[:,1])
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    plt.title(EXPERIMENT_NAME)
    plt.ylim([0.5, 1.0])
    plt.legend()
    plt.grid('on')
    fig.savefig(os.path.join(args['output_dir'],EXPERIMENT_NAME + '_learning_curve.png'))

