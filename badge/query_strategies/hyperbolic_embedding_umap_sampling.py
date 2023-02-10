import numpy as np
import pdb
import torch
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import pairwise_distances
from .strategy import Strategy
import umap
import umap.plot
# hyperparams
from manifolds import Hyperboloid

PROJ_EPS = 1e-3
EPS = 1e-15
MAX_TANH_ARG = 15.0

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


class HypUmapSampleing(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypUmapSampleing, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = Hyperboloid()
        self.curvature = 1/15 # based on plot 4 in HGCN paper

    def query(self, n):
        # TODO: Implement UMAP and Hyperbolic projection
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        # Transform to Hyperboloid model of hyperbolic space
        hypEmbedding = self.manifold.expmap0(embedding, self.curvature)
        # Train UMAP on all hyperbolic embeddings
        # hyperbolic_mapper = umap.UMAP(n_components=10,n_neighbors=5,output_metric='hyperboloid',metric='minkowski',
        #                               random_state=42,tqdm_kwds={'disable':False}).fit(hypEmbedding)
        hyperbolic_mapper_to_euclidean = umap.UMAP(n_components=10, n_neighbors=5, output_metric='euclidean', metric='minkowski',
                                      random_state=42, tqdm_kwds={'disable': False}).fit(hypEmbedding)
        # plt.scatter(hyperbolic_mapper.embedding_.T[0],
        #             hyperbolic_mapper.embedding_.T[1],
        #             c=self.Y, cmap='Spectral')
        # x = hyperbolic_mapper.embedding_[:, 0]
        # y = hyperbolic_mapper.embedding_[:, 1]
        # z = np.sqrt(1 + np.sum(hyperbolic_mapper.embedding_ ** 2, axis=1))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, c=self.Y, cmap='Spectral')
        # ax.view_init(35, 80)
        # plt.show()
        # umap.plot.points(hyperbolic_mapper, labels=self.Y)
        # umap.plot.plt.show()

        # compute hyp-emb for all unlabeled
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # hyperbolic_embedding_unlabeled = hyperbolic_mapper.embedding_[idxs_unlabeled]
        # embedding_unlabeled = self.manifold.logmap0(torch.tensor(hyperbolic_embedding_unlabeled), self.curvature).numpy()



        # Perform UMAP -> transforms to a lower dimension
        # hypUmapEmbedding = UMAP(hypEmbedding)
        # Run Kmeans to get clusters and sample association
        chosen = init_centers(hyperbolic_mapper_to_euclidean.embedding_[idxs_unlabeled], n)
        return idxs_unlabeled[chosen]


        pass

class BaitHypSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaitHypSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        # TODO: see if Fisher transformation makes sense in hyperbolic space
        # compute get_exp_grad for all unlabeled
        pass


class HypNetBadgeSampling(Strategy):
    """
        use hyperbolic layer as last layer,
        use normal cross entropy as BADGE
    """
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypNetBadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        # compute hyp-emb for all unlabeled
        # get_grad
        # Run Kmeans to get clusters and sample association
        # chosen = init_centers(gradEmbedding, n)

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding_for_hyperNet(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = init_centers(gradEmbedding, n)
        return idxs_unlabeled[chosen]



