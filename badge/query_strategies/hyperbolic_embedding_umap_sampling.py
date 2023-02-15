import os

import numpy as np
import pdb

import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import pairwise_distances, adjusted_rand_score, adjusted_mutual_info_score
from .strategy import Strategy
import umap
import umap.plot
# hyperparams
from manifolds import Hyperboloid, PoincareBall
from tqdm import tqdm

from .util import create_directory, plot_clusters_no_edge
from hyperbolic_kmeans.hkmeans import HyperbolicKMeans

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
    # print('#Samps\tTotal Distance')
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
    return indsAll

class UmapHypKmeansSampleing(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UmapHypKmeansSampleing, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = PoincareBall()
        self.curvature = 1/15 # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir,'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    def query(self, n):
        #TODO: get indexes of the chosen samples
        if len(os.listdir(self.output_image_dir)) != 0:
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4])+1
            image_name = os.path.join(self.output_image_dir,"{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir,"chosen_{:05d}.txt".format(name))
            all_sample_name = os.path.join(self.output_sample_dir,"all_{:05d}.txt".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir,'00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir,'chosen_00000.txt')
            all_sample_name = os.path.join(self.output_sample_dir,'all_00000.txt')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        # Run UMAP to reduce dimension
        print('Training UMAP on all samples in Euclidean space ...')
        standard_embedding = umap.UMAP(random_state=42, tqdm_kwds={'disable': False}).fit_transform(embedding)
        # standard_embedding = torch.tensor(standard_embedding)/max(torch.norm(torch.tensor(standard_embedding),dim=1))
        classes = [str(i) for i in range(10)]
        plt.scatter(standard_embedding.T[0], standard_embedding.T[1], c=self.Y, s=2, cmap='Spectral')
        plt.show()
        print('Transform UMAP features to Poincare ball space and normalize in that space ...')
        embedding_poincare = self.manifold.expmap0(torch.tensor(standard_embedding), self.curvature)
        embedding_poincare = (embedding_poincare / max(self.manifold.norm(embedding_poincare))).numpy()
        np.savetxt(all_sample_name, embedding_poincare)

        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Poincare Ball space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        hkmeans = HyperbolicKMeans(n_clusters=10)
        hkmeans.fit(embedding_poincare, max_epochs=15)
        print('Cluster variance: ', hkmeans.variances)
        print('Inertia (poincaré dist): ', hkmeans.inertia_)
        predictions = hkmeans.predict(embedding_poincare)
        predictions_labels = np.array([np.where(predictions[i, :] == 1) for i in range(len(predictions))]).squeeze()

        print(adjusted_rand_score(self.Y, predictions_labels), adjusted_mutual_info_score(self.Y, predictions_labels))
        print('Running Hyperbolic Kmean++ in Poincare Ball space ...')
        # chosen = init_centers(hyperbolic_to_euclidean_emb[idxs_unlabeled], n)
        hyperbolic_emb_chosen = hkmeans.centroids
        np.savetxt(selected_sample_name, np.concatenate([np.expand_dims(idxs_unlabeled[chosen],axis=1),hyperbolic_emb_chosen], axis=1))
        plt.scatter(embedding_poincare.T[0],
                    embedding_poincare.T[1],
                    c=self.Y, cmap='Spectral')
        plt.scatter(hyperbolic_emb_chosen.T[0],
                    hyperbolic_emb_chosen.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        # plot_clusters_no_edge(hyperbolic_to_euclidean_emb, labels, centroids, classes, title=None, height=8, width=8,
        #                       add_labels=False, label_dict=None, plot_frac=1, label_frac=0.001)
        plt.savefig(image_name)
        plt.close('all')
        del hyperbolic_to_euclidean_emb_chosen, hyperbolic_to_euclidean_emb
        return idxs_unlabeled[chosen]


        pass


class HypUmapSampleing(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypUmapSampleing, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = Hyperboloid()
        self.curvature = 1/15 # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir,'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)
        self.classes = [str(i) for i in range(10)]

    def query(self, n):
        if len(os.listdir(self.output_image_dir)) != 0:
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4])+1
            image_name = os.path.join(self.output_image_dir,"{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir,"chosen_{:05d}.txt".format(name))
            all_sample_name = os.path.join(self.output_sample_dir,"all_{:05d}.txt".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir,'00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir,'chosen_00000.txt')
            all_sample_name = os.path.join(self.output_sample_dir,'all_00000.txt')
        # Get embedding for all data
        print('Get embedding for all Samples ...')
        embedding = self.get_embedding(self.X, self.Y)
        # Transform to Hyperboloid model of hyperbolic space
        print('Transform embedding to hyperbolic space using Hyperbloid model ...')
        hypEmbedding = self.manifold.expmap0(embedding, self.curvature)
        # Train UMAP on all hyperbolic embeddings
        # hyperbolic_mapper = umap.UMAP(n_components=10,n_neighbors=5,output_metric='hyperboloid',metric='minkowski',
        #                               random_state=42,tqdm_kwds={'disable':False}).fit(hypEmbedding)
        print('Training UMAP on all samples and returning the embedding in Euclidean space ...')
        hyperbolic_to_euclidean_emb = umap.UMAP(n_components=10, output_metric='euclidean', metric='minkowski',
                                                random_state=42, tqdm_kwds={'disable': False}).fit_transform(hypEmbedding)
        print('Normalize embedding in Euclidean space ...')
        hyperbolic_to_euclidean_emb = hyperbolic_to_euclidean_emb / max(scipy.linalg.norm(hyperbolic_to_euclidean_emb, axis=1))
        np.savetxt(all_sample_name, hyperbolic_to_euclidean_emb)

        # compute hyp-emb for all unlabeled
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # Run Kmeans to get clusters and sample association
        print('Running Kmean++ in Euclidean space ...')
        chosen = init_centers(hyperbolic_to_euclidean_emb[idxs_unlabeled], n)
        hyperbolic_to_euclidean_emb_chosen = hyperbolic_to_euclidean_emb[idxs_unlabeled[chosen]]
        np.savetxt(selected_sample_name, np.concatenate([np.expand_dims(idxs_unlabeled[chosen],axis=1),hyperbolic_to_euclidean_emb_chosen], axis=1))
        plt.scatter(hyperbolic_to_euclidean_emb.T[0],
                    hyperbolic_to_euclidean_emb.T[1],
                    c=self.Y, cmap='Spectral')
        plt.scatter(hyperbolic_to_euclidean_emb_chosen.T[0],
                    hyperbolic_to_euclidean_emb_chosen.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        # emb = np.concatenate(
        #     [np.expand_dims(np.arange(0, len(hyperbolic_to_euclidean_emb)), axis=1), hyperbolic_to_euclidean_emb],
        #     axis=1)[:, 0:3]
        # emb = pd.DataFrame(emb, columns=['node', 'x', 'y'])
        # plot_clusters_no_edge(emb, self.Y, hyperbolic_to_euclidean_emb_chosen[:,0:2], self.classes)
        plt.savefig(image_name)
        plt.close('all')
        del hyperbolic_to_euclidean_emb_chosen, hyperbolic_to_euclidean_emb
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



