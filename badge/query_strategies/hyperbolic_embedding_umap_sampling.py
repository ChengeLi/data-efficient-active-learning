import math
import os
import random

import numpy as np
import pdb
from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
import scipy
import torch
from torch import nn
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import pairwise_distances, adjusted_rand_score, adjusted_mutual_info_score
from .strategy import Strategy
import umap
import umap.plot
# hyperparams
from manifolds import Hyperboloid, PoincareBall
from .util import create_directory, plot_clusters_no_edge
import pdb
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import hyptorch.pmath as pmath
from hyptorch.loss import contrastive_loss
# from apex import amp

import multiprocessing
from hyptorch.sampler import UniqueClassSempler
from torch.autograd import Variable
# from dataset import CUBirds


PROJ_EPS = 1e-3
EPS = 1e-15
MAX_TANH_ARG = 15.0


def entropy_score_np(prob_dist):
    """ Entropy-Based Uncertainty Sampling
    Returns the uncertainty score of a probability distribution using
    entropy score

    Assumes probability distribution is a numpy 1d array like:
        [0.0321, 0.6439, 0.0871, 0.2369]

    Keyword arguments:
        prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
        sorted -- if the probability distribution is pre-sorted from largest to smallest
    """
    log_probs = prob_dist * np.log2(prob_dist)  # multiply each probability by its base 2 log
    raw_entropy = 0 - np.sum(log_probs)

    normalized_entropy = raw_entropy / np.log2(prob_dist.size)

    return normalized_entropy

def entropy_score(prob_dist):
        """
        Returns the uncertainty score of a probability distribution using
        entropy

        Assumes probability distribution is a pytorch tensor, like:
            tensor([0.0321, 0.6439, 0.0871, 0.2369])

        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        log_probs = prob_dist * torch.log2(prob_dist)  # multiply each probability by its base 2 log
        raw_entropy = 0 - torch.sum(log_probs)

        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())

        return normalized_entropy.item()

class BadgePoincareSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgePoincareSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = PoincareBall()
        self.curvature = 1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.argmin([self.manifold.norm(s) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] < newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        if len(os.listdir(self.output_sample_dir)) != 0:
            name = int(sorted(os.listdir(self.output_sample_dir))[-1][4:-4]) + 1
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, "emb_00000.npy")
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        del embedding
        print('Computing gradEmbedding ...')
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])

        print('Transform grad emb to Poincare ball space and normalize in that space ...')
        gradEmbedding_poincare = self.manifold.expmap0(gradEmbedding.clone().detach(), self.curvature)
        gradEmbedding_poincare = (gradEmbedding_poincare / max(self.manifold.norm(gradEmbedding_poincare)))

        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Poincare Ball space ...')
        chosen = self.init_centers_hyp(gradEmbedding_poincare, n)

        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)

        del gradEmbedding_poincare, df, gradEmbedding
        return idxs_unlabeled[chosen]

class PoincareKmeansSamplingNew(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(PoincareKmeansSamplingNew, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = PoincareBall()
        self.curvature = 1/15 #0.03 0.5 #1/10 #1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']

        self.output_image_dir = os.path.join(self.output_dir, 'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_sample_dir)
        create_directory(self.output_image_dir)

    # kmeans ++ initialization
    def init_centers(self, X, K):
        ind = np.argmin(X)
        mu = [int((X[ind]).numpy())]
        indsAll = [int(ind.numpy())]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        # t1 = time()
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(np.expand_dims(X,axis=1), np.array([[mu[-1]]])).ravel().astype(float)
            else:
                newD = pairwise_distances(np.expand_dims(X,axis=1), np.array([[mu[-1]]])).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] < newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            # if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        # t2 = time()
        # print(f'init centers took {t2 - t1} seconds')
        return indsAll
    def init_centers_hyp(self, X, K):
        ind = np.argmin([self.manifold.norm(torch.tensor(s.clone().detach()), self.curvature) for s in X.clone().detach()])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll
    def query(self, n):

        if len(os.listdir(self.output_sample_dir)) != 0:
            name = int(sorted(os.listdir(self.output_sample_dir))[-1][4:-4]) + 1
            image_name = os.path.join(self.output_image_dir, "{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir, '00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        embedding = (embedding / max(torch.norm(embedding,dim=1)))

        print('Transform model emb to Poincare ball space ...')
        all_emb = self.manifold.expmap0(embedding.clone().detach(), self.curvature)
        # all_emb = (all_emb / max(self.manifold.norm(all_emb,self.curvature)))
        all_emb_norm = self.manifold.norm(all_emb,self.curvature)#[self.manifold.norm(s, self.curvature) for s in all_emb]
        # fit unsupervised clusters and plot results
        print('Running Kmean++ on 1D norm ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen_uncertainty = self.init_centers(all_emb_norm[idxs_unlabeled],n-int(3*n/4)+10)#np.argsort(all_emb_norm[idxs_unlabeled].numpy())[:n-int(3*n/4)+10].tolist() #
        chosen_diversity = self.init_centers_hyp(all_emb[idxs_unlabeled], int(3*n/4))
        chosen = random.sample(set(chosen_uncertainty).union(set(chosen_diversity)), n) #np.concatenate([chosen_diversity,chosen_uncertainty]).tolist()#list(set(chosen_uncertainty.tolist()).union(set(chosen_diversity)))
        # print(len(chosen))
        plt.scatter(all_emb_norm[idxs_unlabeled],np.zeros_like(all_emb_norm[idxs_unlabeled]), c=self.Y[idxs_unlabeled], s=2, cmap='Spectral')
        plt.scatter(all_emb_norm[idxs_unlabeled[chosen_uncertainty]], np.zeros_like(all_emb_norm[idxs_unlabeled[chosen_uncertainty]]),
                    c=self.Y[idxs_unlabeled[chosen_uncertainty]], edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.savefig(image_name)
        plt.close('all')
        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)

        del all_emb, df, embedding, all_emb_norm
        return idxs_unlabeled[chosen]


class PoincareKmeansSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(PoincareKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = PoincareBall()
        self.curvature = 1/15 #0.03 0.5 #1/10 #1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.argmin([self.manifold.norm(torch.tensor(s), self.curvature) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):

        if len(os.listdir(self.output_sample_dir)) != 0:
            name = int(sorted(os.listdir(self.output_sample_dir))[-1][4:-4]) + 1
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        embedding = (embedding / max(torch.norm(embedding,dim=1))).clone().detach()

        print('Transform model emb to Poincare ball space and normalize in that space ...')
        all_emb = self.manifold.expmap0(embedding, self.curvature)
        # all_emb = (all_emb / max(self.manifold.norm(all_emb,self.curvature)))
        del embedding
        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Poincare Ball space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers_hyp(all_emb[idxs_unlabeled], n)
        del all_emb
        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)

        del df
        return idxs_unlabeled[chosen]


class HyperboloidKmeansSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HyperboloidKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = Hyperboloid()
        self.curvature = 1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        # self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        # create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.random.choice(np.arange(0, len(X)))  # np.argmin([self.manifold.norm(torch.tensor(s)) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        if len(os.listdir(self.output_sample_dir)) != 0:
            name = int(sorted(os.listdir(self.output_sample_dir))[-1][4:-4]) + 1
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        print('Transform model emb to Poincare ball space and normalize in that space ...')
        all_emb = self.manifold.expmap0(embedding.clone().detach(), self.curvature)
        all_emb = (all_emb / max(self.manifold.norm(all_emb.clone().detach())))

        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Hyperboloid space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers_hyp(all_emb[idxs_unlabeled], n)

        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)
        del embedding, all_emb, df
        return idxs_unlabeled[chosen]


class UmapKmeansSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UmapKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.output_dir = args['output_dir']
        # self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        # create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers(self, X, K):
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
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        # t2 = time()
        # print(f'init centers took {t2 - t1} seconds')
        return indsAll

    def query(self, n):

        if len(os.listdir(self.output_sample_dir)) != 0:
            name = int(sorted(os.listdir(self.output_sample_dir))[-1][4:-4]) + 1
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_sample_name_euclidean = os.path.join(self.output_sample_dir, "all_euclidean_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_sample_name_euclidean = os.path.join(self.output_sample_dir, 'all_euclidean_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))

        # Run UMAP to reduce dimension
        print('Training UMAP on all samples in Euclidean space ...')
        standard_embedding = umap.UMAP(n_components=10, random_state=42, tqdm_kwds={'disable': False}).fit_transform(
            embedding)

        header_ = ['emb_' + str(i) for i in range(np.shape(standard_embedding)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(), axis=1), standard_embedding], axis=1),
                          columns=header_)
        df.to_csv(all_sample_name_euclidean, index=False)

        print('Running Kmean++ in Euclidean space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers(standard_embedding[idxs_unlabeled], n)

        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)

        del standard_embedding, df
        return idxs_unlabeled[chosen]


class UmapPoincareKmeansSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UmapPoincareKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = PoincareBall()
        self.curvature = 1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir, 'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.argmin([self.manifold.norm(torch.tensor(s)) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        if len(os.listdir(self.output_image_dir)) != 0:
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4]) + 1
            image_name = os.path.join(self.output_image_dir, "{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_sample_name_poincare = os.path.join(self.output_sample_dir, "all_poincare_{:05d}.csv".format(name))
            all_sample_name_euclidean = os.path.join(self.output_sample_dir, "all_euclidean_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir, '00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_sample_name_poincare = os.path.join(self.output_sample_dir, 'all_poincare_00000.csv')
            all_sample_name_euclidean = os.path.join(self.output_sample_dir, 'all_euclidean_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))

        # Run UMAP to reduce dimension
        print('Training UMAP on all samples in Euclidean space ...')
        standard_embedding = umap.UMAP(random_state=42, tqdm_kwds={'disable': False}).fit_transform(embedding)
        ## standard_embedding = torch.tensor(standard_embedding)/max(torch.norm(torch.tensor(standard_embedding),dim=1))
        # classes = [str(i) for i in range(10)]
        # plt.scatter(standard_embedding.T[0], standard_embedding.T[1], c=self.Y, s=2, cmap='Spectral')
        # plt.show()
        header_ = ['emb_' + str(i) for i in range(np.shape(standard_embedding)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(), axis=1), standard_embedding], axis=1),
                          columns=header_)
        df.to_csv(all_sample_name_euclidean, index=False)
        print('Transform UMAP features to Poincare ball space and normalize in that space ...')
        all_emb = self.manifold.expmap0(torch.tensor(standard_embedding), self.curvature)
        all_emb = (all_emb / max(self.manifold.norm(all_emb)))
        # plt.scatter(all_emb.T[0], all_emb.T[1], c=self.Y, s=2, cmap='Spectral')
        # plt.show()
        header_ = ['emb_' + str(i) for i in range(np.shape(all_emb)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(), axis=1), all_emb.numpy()], axis=1),
                          columns=header_)
        df.to_csv(all_sample_name_poincare, index=False)
        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Poincare Ball space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers_hyp(all_emb[idxs_unlabeled], n)
        chosen_emb = all_emb[idxs_unlabeled[chosen]]

        header_ = ['emb_' + str(i) for i in range(np.shape(chosen_emb)[1])]
        header_ = ['label', 'index'] + header_
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1),
             chosen_emb.numpy()], axis=1), columns=header_)
        df.to_csv(selected_sample_name, index=False)
        plt.scatter(all_emb.T[0],
                    all_emb.T[1],
                    c=self.Y, s=2, cmap='Spectral')
        plt.scatter(chosen_emb.T[0],
                    chosen_emb.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # emb = np.concatenate(
        #     [np.expand_dims(np.arange(0, len(all_emb)), axis=1), all_emb],
        #     axis=1)[:, 0:3]
        # emb = pd.DataFrame(emb, columns=['node', 'x', 'y'])
        # plot_clusters_no_edge(emb, self.Y, chosen_emb[:,0:2], self.classes)
        plt.savefig(image_name)
        plt.close('all')
        del chosen_emb, all_emb, df
        return idxs_unlabeled[chosen]


class UmapHyperboloidKmeansSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UmapHyperboloidKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = Hyperboloid()
        self.curvature = 1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir, 'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.random.choice(np.arange(0, len(X)))  # np.argmin([self.manifold.norm(torch.tensor(s)) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        if len(os.listdir(self.output_image_dir)) != 0:
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4]) + 1
            image_name = os.path.join(self.output_image_dir, "{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_sample_name = os.path.join(self.output_sample_dir, "all_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir, '00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_sample_name = os.path.join(self.output_sample_dir, 'all_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        # Run UMAP to reduce dimension
        print('Training UMAP on all samples in Euclidean space and returning embeddings in hyperboloid space ...')
        all_emb = umap.UMAP(random_state=42, output_metric='hyperboloid', tqdm_kwds={'disable': False}).fit_transform(
            embedding)
        print('Transform UMAP features to Poincare ball space and normalize in that space ...')
        # all_emb = self.manifold.expmap0(torch.tensor(standard_embedding), self.curvature)
        all_emb = (all_emb / max(self.manifold.norm(torch.tensor(all_emb))))
        # plt.scatter(all_emb.T[0], all_emb.T[1], c=self.Y, s=2, cmap='Spectral')
        # plt.show()
        header_ = ['emb_' + str(i) for i in range(np.shape(all_emb)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(), axis=1), all_emb.numpy()], axis=1),
                          columns=header_)
        df.to_csv(all_sample_name, index=False)

        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Hyperboloid space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers_hyp(all_emb[idxs_unlabeled], n)
        chosen_emb = all_emb[idxs_unlabeled[chosen]]

        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)
        plt.scatter(all_emb.T[0],
                    all_emb.T[1],
                    c=self.Y, s=2, cmap='Spectral')
        plt.scatter(chosen_emb.T[0],
                    chosen_emb.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # emb = np.concatenate(
        #     [np.expand_dims(np.arange(0, len(all_emb)), axis=1), all_emb],
        #     axis=1)[:, 0:3]
        # emb = pd.DataFrame(emb, columns=['node', 'x', 'y'])
        # plot_clusters_no_edge(emb, self.Y, chosen_emb[:,0:2], self.classes)
        plt.savefig(image_name)
        plt.close('all')
        del embedding, chosen_emb, all_emb, df
        return idxs_unlabeled[chosen]
        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)
        plt.scatter(all_emb.T[0],
                    all_emb.T[1],
                    c=self.Y, s=2, cmap='Spectral')
        plt.scatter(chosen_emb.T[0],
                    chosen_emb.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # emb = np.concatenate(
        #     [np.expand_dims(np.arange(0, len(all_emb)), axis=1), all_emb],
        #     axis=1)[:, 0:3]
        # emb = pd.DataFrame(emb, columns=['node', 'x', 'y'])
        # plot_clusters_no_edge(emb, self.Y, chosen_emb[:,0:2], self.classes)
        plt.savefig(image_name)
        plt.close('all')
        del embedding, chosen_emb, all_emb, df
        return idxs_unlabeled[chosen]
        pass


class UmapHyperboloidKmeansSampling2(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UmapHyperboloidKmeansSampling2, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = Hyperboloid()
        self.curvature = 1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir, 'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.random.choice(np.arange(0, len(X)))  # np.argmin([self.manifold.norm(torch.tensor(s)) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        if len(os.listdir(self.output_image_dir)) != 0:
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4]) + 1
            image_name = os.path.join(self.output_image_dir, "{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_sample_name = os.path.join(self.output_sample_dir, "all_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir, '00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_sample_name = os.path.join(self.output_sample_dir, 'all_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        # Run UMAP to reduce dimension
        print('Training UMAP on all samples in Euclidean space  ...')
        all_emb = umap.UMAP(random_state=42, tqdm_kwds={'disable': False}).fit_transform(embedding)
        print('Transform UMAP features to Poincare ball space and normalize in that space ...')
        all_emb = self.manifold.expmap0(torch.tensor(all_emb), self.curvature)
        all_emb = (all_emb / max(self.manifold.norm(torch.tensor(all_emb))))
        # plt.scatter(all_emb.T[0], all_emb.T[1], c=self.Y, s=2, cmap='Spectral')
        # plt.show()
        header_ = ['emb_' + str(i) for i in range(np.shape(all_emb)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(), axis=1), all_emb.numpy()], axis=1),
                          columns=header_)
        df.to_csv(all_sample_name, index=False)

        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Hyperboloid space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers_hyp(all_emb[idxs_unlabeled], n)
        chosen_emb = all_emb[idxs_unlabeled[chosen]]

        header_ = ['label', 'index']
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1),
            columns=header_)
        df.to_csv(selected_sample_name, index=False)
        plt.scatter(all_emb.T[0],
                    all_emb.T[1],
                    c=self.Y, s=2, cmap='Spectral')
        plt.scatter(chosen_emb.T[0],
                    chosen_emb.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # emb = np.concatenate(
        #     [np.expand_dims(np.arange(0, len(all_emb)), axis=1), all_emb],
        #     axis=1)[:, 0:3]
        # emb = pd.DataFrame(emb, columns=['node', 'x', 'y'])
        # plot_clusters_no_edge(emb, self.Y, chosen_emb[:,0:2], self.classes)
        plt.savefig(image_name)
        plt.close('all')
        del embedding, chosen_emb, all_emb, df
        return idxs_unlabeled[chosen]
        pass


class HypUmapSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypUmapSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = Hyperboloid()
        self.curvature = 1 / 15  # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir, 'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)
        self.classes = [str(i) for i in range(10)]

    # kmeans ++ initialization
    def init_centers(self, X, K):
        ind = np.random.choice(np.arange(0, len(X)))
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()\
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        if len(os.listdir(self.output_image_dir)) != 0:
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4]) + 1
            image_name = os.path.join(self.output_image_dir, "{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir, "chosen_{:05d}.csv".format(name))
            all_sample_name = os.path.join(self.output_sample_dir, "all_{:05d}.csv".format(name))
            all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir, '00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir, 'chosen_00000.csv')
            all_sample_name = os.path.join(self.output_sample_dir, 'all_00000.csv')
            all_emb_name = os.path.join(self.output_sample_dir, "emb_00000.npy")
        # Get embedding for all data
        print('Get embedding for all Samples ...')
        embedding = self.get_embedding(self.X, self.Y)
        np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1), embedding], axis=1))
        # Transform to Hyperboloid model of hyperbolic space
        print('Transform embedding to hyperbolic space using Hyperbloid model ...')
        hypEmbedding = self.manifold.expmap0(embedding, self.curvature)
        # Train UMAP on all hyperbolic embeddings
        # hyperbolic_mapper = umap.UMAP(n_components=10,n_neighbors=5,output_metric='hyperboloid',metric='minkowski',
        #                               random_state=42,tqdm_kwds={'disable':False}).fit(hypEmbedding)
        print('Training UMAP on all samples and returning the embedding in Euclidean space ...')
        all_emb = umap.UMAP(output_metric='euclidean', metric='minkowski',
                            random_state=42, tqdm_kwds={'disable': False}).fit_transform(hypEmbedding)
        print('Normalize embedding in Euclidean space ...')
        all_emb = all_emb / max(scipy.linalg.norm(all_emb, axis=1))
        header_ = ['emb_' + str(i) for i in range(np.shape(all_emb)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(), axis=1), all_emb], axis=1), columns=header_)
        df.to_csv(all_sample_name, index=False)
        # compute hyp-emb for all unlabeled
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # Run Kmeans to get clusters and sample association
        print('Running Kmean++ in Euclidean space ...')
        chosen = self.init_centers(all_emb[idxs_unlabeled], n)
        chosen_emb = all_emb[idxs_unlabeled[chosen]]
        header_ = ['emb_' + str(i) for i in range(np.shape(chosen_emb)[1])]
        header_ = ['label', 'index'] + header_
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1),
             chosen_emb], axis=1), columns=header_)
        df.to_csv(selected_sample_name, index=False)
        plt.scatter(all_emb.T[0],
                    all_emb.T[1],
                    c=self.Y, cmap='Spectral')
        plt.scatter(chosen_emb.T[0],
                    chosen_emb.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # emb = np.concatenate(
        #     [np.expand_dims(np.arange(0, len(all_emb)), axis=1), all_emb],
        #     axis=1)[:, 0:3]
        # emb = pd.DataFrame(emb, columns=['node', 'x', 'y'])
        # plot_clusters_no_edge(emb, self.Y, chosen_emb[:,0:2], self.classes)
        plt.savefig(image_name)
        plt.close('all')
        del embedding, chosen_emb, all_emb, df
        return idxs_unlabeled[chosen]

        pass


class HypNetBadgeSampling(Strategy):
    """
        use hyperbolic layer as last layer,
        use normal cross entropy as BADGE
    """
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypNetBadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.curvature = args['poincare_ball_curvature']
        self.radius = 1

        self.manifold = PoincareBall()
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)
        self.iteration_ind = 0

    def init_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])  # this only make sense for badge
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                try:
                    D2 = pairwise_distances(X, mu).ravel().astype(float)
                except:
                    pdb.set_trace()
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll


    def query(self, n):
        """
            option 1: use regular gradient in badge
            option 2: fix grad using riemannian gradient in badge
            
        """
        use_Riemannian_grad_badge = False

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding, embedding = self.get_grad_embedding_for_hyperNet(self.X, self.Y, 
                                                             fix_grad=use_Riemannian_grad_badge,
                                                             RiemannianGradient_c=self.curvature,
                                                             get_raw_embedding=True,
                                                             back_to_euclidean=True)
        chosen = self.init_centers(gradEmbedding[idxs_unlabeled], n)
        ## normalizae embedding
        # embedding = (embedding / max(self.manifold.norm(torch.tensor(embedding), self.curvature)))
        embedding = embedding / max([np.linalg.norm(s, 2) for s in embedding])
        self.save_images_and_embeddings(embedding, idxs_unlabeled, chosen, self.iteration_ind)
        self.iteration_ind += 1
        return idxs_unlabeled[chosen]



class HypNetNormSampling(Strategy):
    """
        use hyperbolic embedding's norm as a measure of uncertainty
    """

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypNetNormSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = PoincareBall()
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)
        self.curvature = args['poincare_ball_curvature']
        self.iteration_ind = 0

    def init_centers(self, X, K):
        # hyper_emb_norm = [np.linalg.norm(s, 2) for s in X]
        hyper_emb_norm = [self.manifold.norm(torch.tensor(s), self.curvature) for s in X]
        indsAll = np.argsort(hyper_emb_norm, axis=0)[:K]  # select the smallest K samples
        return indsAll

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embedding = self.get_hyperbolic_embedding(self.X, self.Y).numpy()
        chosen = self.init_centers(embedding[idxs_unlabeled], n)

        ## normalizae embedding
        # embedding = (embedding / max(self.manifold.norm(torch.tensor(embedding), self.curvature)))
        # embedding = embedding / max([np.linalg.norm(s, 2) for s in embedding])
        self.save_images_and_embeddings(embedding, idxs_unlabeled, chosen, self.iteration_ind)
        self.iteration_ind += 1
        return idxs_unlabeled[chosen]


    # def train(self, reset=True, optimizer=0, verbose=True, data=[], net=[], model_selection=None):
    #     ## override the train function
    #     def weight_reset(m):
    #         newLayer = deepcopy(m)
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             m.reset_parameters()
    #     n_epoch = self.args['n_epoch']
    #     if reset: self.clf =  self.net.apply(weight_reset).cuda()
    #     if type(net) != list: self.clf = net
    #     if type(optimizer) == int: optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

    #     idxs_train = np.arange(self.n_pool)[self.idxs_lb]
    #     # loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

    #     # data_root = '/workplace/ICCV_AL/data/' #images/
    #     # ds_train = CUBirds(data_root, "train", transform=self.args['transform'])

    #     ## configs
    #     # for mnist:
    #     num_classes = 10
    #     num_samples = self.args['loader_tr_args']['batch_size']//num_classes #2 # how many samples per each category in batch
    #     self.args['loader_tr_args']['batch_size'] = num_samples*num_classes
    #     print('changing batch_size to ', self.args['loader_tr_args']['batch_size'])
    #     ## for CUB
    #     # num_samples = 2

    #     t = 0.2  # cross-entropy temperature
    #     emb = self.args['poincare_ball_dim']#128  # output embedding size
    #     local_rank = 0  # set automatically for distributed training
    #     world_size = int(os.environ.get("WORLD_SIZE", 1))


    #     # sampler = UniqueClassSempler(
    #     #     ds_train.ys, num_samples, local_rank, world_size
    #     # )

    #     # dl_train = DataLoader(
    #     #     dataset=ds_train,
    #     #     sampler=sampler,
    #     #     batch_size=self.args['loader_tr_args']['batch_size'],
    #     # )

    #     print('len(idxs_train)={}'.format(len(idxs_train)))
    #     sampler2 = UniqueClassSempler(
    #         self.Y.numpy()[idxs_train], num_samples, local_rank, world_size
    #     )

    #     dl_train2 = DataLoader(
    #                 self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
    #                 transform=self.args['transform']),
    #                 sampler=sampler2, **self.args['loader_tr_args']
    #         )

    #     cuda_ind = 0 ##which gpu to use
    #     # epoch, attempts (early stopping), training accuracy
    #     if verbose: print('epoch, attempts, training accuracy:, acc', flush=True)
    #     epoch = 0
    #     bestAcc = 0.
    #     attempts = 0
    #     accCurrent = 0
    #     while epoch < self.args['max_epoch'] and accCurrent < 0.99: #train for max_epoch epoches at most
    #         if not isinstance(self.clf, nn.DataParallel):
    #             print('enabling multiple gpus')
    #             self.clf = self.enable_multiple_gpu_model(self.clf)

    #         self.clf.train()
    #         sampler2.set_epoch(epoch)
    #         stats_ep = []
    #         accCurrent = 0.
    #         num_samples_runned = 0
    #         for batch_idx, (x, y, idxs) in enumerate(dl_train2):
    #             # print(x[0].sum(), x[1].sum(), x[2].sum(),x[3].sum())
    #             # print(x[0].mean(), x[1].mean(), x[2].mean(),x[3].mean())
    #             # print(y)
    #             # print('idxs=',idxs)
    #             x, y = Variable(x.to(torch.device(f"cuda:{cuda_ind}"))), Variable(y.to(torch.device(f"cuda:{cuda_ind}")))
    #             orgional_y = deepcopy(y)
    #             y = y.view(len(y) // num_samples, num_samples)
    #             assert (y[:, 0] == y[:, -1]).all()
    #             s = y[:, 0].tolist()
    #             assert len(set(s)) == len(s)

    #             out, e1 = self.clf(x)
    #             z = e1.view(len(x) // num_samples, num_samples, emb).to(torch.device(f"cuda:{cuda_ind}"))
    #             if world_size > 1:
    #                 with torch.no_grad():
    #                     all_z = [torch.zeros_like(z) for _ in range(world_size)]
    #                     torch.distributed.all_gather(all_z, z)
    #                 all_z[local_rank] = z
    #                 z = torch.cat(all_z)
    #             loss = 0
    #             for i in range(num_samples):
    #                 for j in range(num_samples):
    #                     if i != j:
    #                         l, s = contrastive_loss(z[:, i], z[:, j], target=y,
    #                                 tau=t, hyp_c=self.args['poincare_ball_curvature'], 
    #                                 cuda_ind=cuda_ind)
    #                         loss += l
    #                         stats_ep.append({**s, "loss": l.item()})

    #             # loss /= num_samples*num_samples-num_samples
    #             # ce_loss = F.cross_entropy(out.to(torch.device(f"cuda:{cuda_ind}")), orgional_y)
    #             ce_loss = 0
    #             loss += ce_loss
    #             print('loss', loss, ce_loss)
    #             optimizer.zero_grad()
    #             # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #             #     scaled_loss.backward()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.clf.parameters(), 3)
    #             optimizer.step()

    #             batchProbs = F.softmax(out, dim=1)
    #             maxInds = torch.argmax(batchProbs,1).data.cpu()
    #             accCurrent += torch.sum(maxInds == orgional_y.data.cpu()).float().data.item()
    #             num_samples_runned += len(orgional_y)

    #         accCurrent /= num_samples_runned
    #         if bestAcc < accCurrent:
    #             bestAcc = accCurrent
    #             attempts = 0
    #         else: attempts += 1
    #         epoch += 1
    #         if verbose: print(str(epoch) + '_' + str(attempts) + ' training accuracy: ' + str(accCurrent), flush=True)
    #         # reset if not converging
    #         if (epoch % 1000 == 0) and (accCurrent < 0.2) and (self.args['modelType'] != 'linear'):
    #             self.clf = self.net.apply(weight_reset)
    #             optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)



class HyperNorm_plus_RiemannianBadge_Sampling(Strategy):
    """
        use a combination of hyperbolic embedding's norm
        and Riemannian badge (badge with fix grad)

    """
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HyperNorm_plus_RiemannianBadge_Sampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.HypNetNormSampler = HypNetNormSampling(X, Y, idxs_lb, net, handler, args)
        self.HypNetBadgeSampler = HypNetBadgeSampling(X, Y, idxs_lb, net, handler, args)

    def train(self, reset=True, optimizer=0, verbose=True, data=[], net=[], model_selection=None):       
        super(HyperNorm_plus_RiemannianBadge_Sampling, self).train(reset, optimizer, verbose, data, net, model_selection)
        ### init the sampling strategy's clf due to the train() is not called for them
        self.HypNetNormSampler.clf = self.clf
        self.HypNetBadgeSampler.clf = self.clf

 
    def query(self, n):
        self.HypNetNormSampler.update(self.idxs_lb)
        ids_selected_by_norm = self.HypNetNormSampler.query(n//2)

        self.idxs_lb[ids_selected_by_norm] = True
        self.HypNetBadgeSampler.update(self.idxs_lb)
        ids_selected_by_RieBadge = self.HypNetBadgeSampler.query(n-n//2)

        self.idxs_lb[ids_selected_by_RieBadge] = True
        selected_inds = set(ids_selected_by_norm).union(set(ids_selected_by_RieBadge))
        assert len(selected_inds)==n
        return [ii for ii in selected_inds]


class HypNetBadgePoincareKmeansSampling(Strategy):
    """
        use hyperbolic layer as last layer,
        use normal cross entropy as BADGE.
        use Poincare Kmeans
    """
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypNetBadgePoincareKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.curvature = args['poincare_ball_curvature']
        self.manifold = PoincareBall()

        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)
        self.iteration_ind = 0

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        ## use the real norm to get the max gradient one
        # hyper_emb_norm = [self.manifold.norm(torch.tensor(s), self.curvature) for s in X]
        # ind = np.argmax(hyper_emb_norm, axis=0)
        # print(f'ind0={ind0}, ind={ind}') # they are the same

        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        """
            option 1: use regular gradient in badge
            option 2: fix grad using riemannian gradient in badge
       """
        use_Riemannian_grad_badge = True

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        gradEmbedding, embedding = self.get_grad_embedding_for_hyperNet(self.X, self.Y.numpy(), 
                                                             fix_grad=use_Riemannian_grad_badge,
                                                             RiemannianGradient_c=self.curvature,
                                                             get_raw_embedding=True,
                                                             back_to_euclidean=False)
        chosen = self.init_centers_hyp(torch.tensor(gradEmbedding[idxs_unlabeled]), n)
        ## old normalization
        # embedding = embedding / max([np.linalg.norm(s, 2) for s in embedding])
        self.save_images_and_embeddings(embedding, idxs_unlabeled, chosen, self.iteration_ind)
        self.iteration_ind += 1
        return idxs_unlabeled[chosen]


class HypNetEmbeddingPoincareKmeansSampling(Strategy):
    """
        use hyperbolic layer as last layer,
        use normal cross entropy as BADGE.
        use Poincare Kmeans
    """
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypNetEmbeddingPoincareKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.curvature = args['poincare_ball_curvature']
        self.manifold = PoincareBall()

        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir, 'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)
        self.iteration_ind = 0

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
            else:
                newD = self.manifold.sqdist(X, mu[-1], self.curvature).ravel().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def query(self, n):
        """
            option 1: use regular gradient in badge
            option 2: fix grad using riemannian gradient in badge
       """
        use_Riemannian_grad_badge = True

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        embedding = self.get_hyperbolic_embedding(self.X, self.Y).numpy()
        chosen = self.init_centers_hyp(torch.tensor(embedding[idxs_unlabeled]), n) #note the only difference with hyperBadge
        self.save_images_and_embeddings(embedding, idxs_unlabeled, chosen, self.iteration_ind)
        self.iteration_ind += 1
        return idxs_unlabeled[chosen]



# class UmapKmeansSampling(Strategy):
#     def __init__(self, X, Y, idxs_lb, net, handler, args):
#         super(UmapKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
#         self.output_dir = args['output_dir']
#         # self.output_image_dir = os.path.join(self.output_dir,'images')
#         self.output_sample_dir = os.path.join(self.output_dir,'samples')
#         # create_directory(self.output_image_dir)
#         create_directory(self.output_sample_dir)
#
#     # kmeans ++ initialization
#     def init_centers(self, X, K):
#         ind = np.argmax([np.linalg.norm(s, 2) for s in X])
#         mu = [X[ind]]
#         indsAll = [ind]
#         centInds = [0.] * len(X)
#         cent = 0
#         # print('#Samps\tTotal Distance')
#         # t1 = time()
#         while len(mu) < K:
#             if len(mu) == 1:
#                 D2 = pairwise_distances(X, mu).ravel().astype(float)
#             else:
#                 newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
#                 for i in range(len(X)):
#                     if D2[i] > newD[i]:
#                         centInds[i] = cent
#                         D2[i] = newD[i]
#             # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
#             #if sum(D2) == 0.0: pdb.set_trace()
#             D2 = D2.ravel().astype(float)
#             Ddist = (D2 ** 2) / sum(D2 ** 2)
#             customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
#             ind = customDist.rvs(size=1)[0]
#             while ind in indsAll: ind = customDist.rvs(size=1)[0]
#             mu.append(X[ind])
#             indsAll.append(ind)
#             cent += 1
#         # t2 = time()
#         # print(f'init centers took {t2 - t1} seconds')
#         return indsAll
#     def query(self, n):
#
#         if len(os.listdir(self.output_sample_dir)) != 0:
#             name = int(sorted(os.listdir(self.output_sample_dir))[-1][4:-4])+1
#             selected_sample_name = os.path.join(self.output_sample_dir,"chosen_{:05d}.csv".format(name))
#             all_sample_name_euclidean = os.path.join(self.output_sample_dir,"all_euclidean_{:05d}.csv".format(name))
#             all_emb_name = os.path.join(self.output_sample_dir, "emb_{:05d}.npy".format(name))
#             del name
#         else:
#             selected_sample_name = os.path.join(self.output_sample_dir,'chosen_00000.csv')
#             all_sample_name_euclidean = os.path.join(self.output_sample_dir,'all_euclidean_00000.csv')
#             all_emb_name = os.path.join(self.output_sample_dir, 'emb_00000.npy')
#         # Get embedding for all data
#         embedding = self.get_embedding(self.X, self.Y)
#         np.save(all_emb_name, np.concatenate([np.expand_dims(self.Y, axis=1),embedding], axis=1))
#
#         # Run UMAP to reduce dimension
#         print('Training UMAP on all samples in Euclidean space ...')
#         standard_embedding = umap.UMAP(n_components=10, random_state=42, tqdm_kwds={'disable': False}).fit_transform(embedding)
#
#         header_ = ['emb_' + str(i) for i in range(np.shape(standard_embedding)[1])]
#         header_ = ['label'] + header_
#         df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(),axis=1), standard_embedding],axis=1), columns=header_)
#         df.to_csv(all_sample_name_euclidean,index=False)
#
#         print('Running Kmean++ in Euclidean space ...')
#         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
#         chosen = self.init_centers(standard_embedding[idxs_unlabeled], n)
#
#         header_ = ['label', 'index']
#         df = pd.DataFrame(np.concatenate(
#             [np.expand_dims((self.Y[idxs_unlabeled[chosen]]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1)], axis=1), columns=header_)
#         df.to_csv(selected_sample_name, index=False)
#
#         del standard_embedding, df
#         return idxs_unlabeled[chosen]
