# import numpy as np
# from torch.utils.data import DataLoader
import os

import pandas as pd

from .strategy import Strategy
# import pickle
# from scipy.spatial.distance import cosine
# import sys
# import gc
# from scipy.linalg import det
# from scipy.linalg import pinv as inv
# from copy import copy as copy
# from copy import deepcopy as deepcopy
# import torch
# from torch import nn
# import torchfile
# from torch.autograd import Variable
# import resnet
# import vgg
# import torch.optim as optim
import pdb
# from torch.nn import functional as F
# import argparse
# import torch.nn as nn
# from collections import OrderedDict
from scipy import stats
import numpy as np
# import scipy.sparse as sp
# from itertools import product
# from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import pairwise_distances_argmin_min
# from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
# from sklearn.utils.sparsefuncs_fast import assign_rows_csr
# from sklearn.utils.sparsefuncs import mean_variance_axis
# from sklearn.utils.validation import _num_samples
# from sklearn.utils import check_array
# from sklearn.utils import gen_batches
# from sklearn.utils import check_random_state
# from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.validation import FLOAT_DTYPES
# from sklearn.metrics.pairwise import rbf_kernel as rbf
# #from sklearn.externals.six import string_types
# from sklearn.exceptions import ConvergenceWarning

from sklearnex import patch_sklearn

from .util import create_directory

patch_sklearn()
from sklearn.metrics import pairwise_distances
import pdb
from time import time

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    # t1 = time()
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    # t2 = time()
    # print(f'init centers took {t2-t1} seconds')
    return indsAll

class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.output_dir = args['output_dir']
        self.output_sample_dir = os.path.join(self.output_dir,'samples')
        create_directory(self.output_sample_dir)

    def query(self, n):
        if len(os.listdir(self.output_sample_dir)) != 0:
            name = int(sorted(os.listdir(self.output_sample_dir))[-1][4:-4])+1
            selected_sample_name = os.path.join(self.output_sample_dir,"chosen_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            selected_sample_name = os.path.join(self.output_sample_dir,'chosen_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, "emb_00000.npy")
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        print('Computing gradEmbedding ...')
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()

        print('Running Kmean++ ...')
        chosen = init_centers(gradEmbedding, n)

        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[chosen]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1), columns=header_)
        df.to_csv(selected_sample_name, index=False)
        del embedding, gradEmbedding
        return idxs_unlabeled[chosen]
