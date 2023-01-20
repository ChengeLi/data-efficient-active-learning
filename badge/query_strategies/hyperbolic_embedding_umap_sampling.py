import numpy as np
import pdb
import torch
from scipy import stats
from sklearn.metrics import pairwise_distances

from .strategy import Strategy

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def hyp_transformation(emb):
    pass

class HypUmapSampleing(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypUmapSampleing, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        # TODO: Implement UMAP and Hyperbolic projection
        # compute hyp-emb for all unlabeled
        # Perform UMAP -> transforms to a lower dimension
        # Run Kmeans to get clusters and sample association
        # chosen = init_centers(gradEmbedding, n)


        pass

class BaitHypSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaitHypSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        # TODO: Implement UMAP and Hyperbolic projection
        # compute geg_exp_grad for all unlabeled
        # Perform hyp -> transforms to a hyperbolic embedding
        # Run Kmeans to get clusters and sample association
        # chosen = init_centers(gradEmbedding, n)


        pass

class HypBadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypBadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        # TODO: Implement UMAP and Hyperbolic projection
        # compute hyp-emb for all unlabeled
        # get_grad
        # Run Kmeans to get clusters and sample association
        # chosen = init_centers(gradEmbedding, n)


        pass