import os

import numpy as np
import pdb
from sklearnex import patch_sklearn
patch_sklearn()
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
from .util import create_directory, plot_clusters_no_edge


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

class UmapHypKmeansSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UmapHypKmeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = PoincareBall()
        self.curvature = 1/15 # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir,'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.argmax([self.manifold.norm(torch.tensor(s)) for s in X])
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
            if sum(D2) == 0.0: pdb.set_trace()
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
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4])+1
            image_name = os.path.join(self.output_image_dir,"{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir,"chosen_{:05d}.csv".format(name))
            all_sample_name_poincare = os.path.join(self.output_sample_dir,"all_poincare_{:05d}.csv".format(name))
            all_sample_name_euclidean = os.path.join(self.output_sample_dir,"all_euclidean_{:05d}.csv".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir,'00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir,'chosen_00000.csv')
            all_sample_name_poincare = os.path.join(self.output_sample_dir,'all_poincare_00000.csv')
            all_sample_name_euclidean = os.path.join(self.output_sample_dir,'all_euclidean_00000.csv')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        # Run UMAP to reduce dimension
        print('Training UMAP on all samples in Euclidean space ...')
        standard_embedding = umap.UMAP(random_state=42, tqdm_kwds={'disable': False}).fit_transform(embedding)
        ## standard_embedding = torch.tensor(standard_embedding)/max(torch.norm(torch.tensor(standard_embedding),dim=1))
        # classes = [str(i) for i in range(10)]
        # plt.scatter(standard_embedding.T[0], standard_embedding.T[1], c=self.Y, s=2, cmap='Spectral')
        # plt.show()
        header_ = ['emb_' + str(i) for i in range(np.shape(standard_embedding)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(),axis=1), standard_embedding],axis=1), columns=header_)
        df.to_csv(all_sample_name_euclidean,index=False)
        print('Transform UMAP features to Poincare ball space and normalize in that space ...')
        all_emb = self.manifold.expmap0(torch.tensor(standard_embedding), self.curvature)
        all_emb = (all_emb / max(self.manifold.norm(all_emb)))
        # plt.scatter(all_emb.T[0], all_emb.T[1], c=self.Y, s=2, cmap='Spectral')
        # plt.show()
        header_ = ['emb_' + str(i) for i in range(np.shape(all_emb)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(),axis=1), all_emb.numpy()],axis=1), columns=header_)
        df.to_csv(all_sample_name_poincare,index=False)

        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Poincare Ball space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers_hyp(all_emb[idxs_unlabeled], n)
        chosen_emb = all_emb[idxs_unlabeled[chosen]]

        header_ = ['emb_' + str(i) for i in range(np.shape(chosen_emb)[1])]
        header_ = ['label', 'index'] + header_
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[chosen]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1),
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


        pass


class UmapHypKmeansSampling2(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(UmapHypKmeansSampling2, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.manifold = Hyperboloid()
        self.curvature = 1/15 # based on plot 4 in HGCN paper
        self.output_dir = args['output_dir']
        self.output_image_dir = os.path.join(self.output_dir,'images')
        self.output_sample_dir = os.path.join(self.output_dir,'samples')
        create_directory(self.output_image_dir)
        create_directory(self.output_sample_dir)

    # kmeans ++ initialization
    def init_centers_hyp(self, X, K):
        ind = np.argmax([self.manifold.norm(torch.tensor(s)) for s in X])
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
            if sum(D2) == 0.0: pdb.set_trace()
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
            name = int(sorted(os.listdir(self.output_image_dir))[-1][:-4])+1
            image_name = os.path.join(self.output_image_dir,"{:05d}.png".format(name))
            selected_sample_name = os.path.join(self.output_sample_dir,"chosen_{:05d}.csv".format(name))
            all_sample_name = os.path.join(self.output_sample_dir,"all_{:05d}.csv".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir,'00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir,'chosen_00000.csv')
            all_sample_name = os.path.join(self.output_sample_dir,'all_00000.csv')
        # Get embedding for all data
        embedding = self.get_embedding(self.X, self.Y)
        # Run UMAP to reduce dimension
        print('Training UMAP on all samples in Euclidean space and returning embeddings in hyperboloid space ...')
        all_emb = umap.UMAP(random_state=42, output_metric='hyperboloid', tqdm_kwds={'disable': False}).fit_transform(embedding)
        print('Transform UMAP features to Poincare ball space and normalize in that space ...')
        # all_emb = self.manifold.expmap0(torch.tensor(standard_embedding), self.curvature)
        all_emb = (all_emb / max(self.manifold.norm(torch.tensor(all_emb))))
        plt.scatter(all_emb.T[0], all_emb.T[1], c=self.Y, s=2, cmap='Spectral')
        plt.show()
        header_ = ['emb_' + str(i) for i in range(np.shape(all_emb)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(),axis=1), all_emb.numpy()],axis=1), columns=header_)
        df.to_csv(all_sample_name,index=False)

        # fit unsupervised clusters and plot results
        print('Running Hyperbolic Kmean++ in Poincare Ball space ...')
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        chosen = self.init_centers_hyp(all_emb[idxs_unlabeled], n)
        chosen_emb = all_emb[idxs_unlabeled[chosen]]

        header_ = ['emb_' + str(i) for i in range(np.shape(chosen_emb)[1])]
        header_ = ['label', 'index'] + header_
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[chosen]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1),
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
        pass


class HypUmapSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(HypUmapSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
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
            selected_sample_name = os.path.join(self.output_sample_dir,"chosen_{:05d}.csv".format(name))
            all_sample_name = os.path.join(self.output_sample_dir,"all_{:05d}.csv".format(name))
            del name
        else:
            image_name = os.path.join(self.output_image_dir,'00000.png')
            selected_sample_name = os.path.join(self.output_sample_dir,'chosen_00000.csv')
            all_sample_name = os.path.join(self.output_sample_dir,'all_00000.csv')
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
        all_emb = umap.UMAP(n_components=10, output_metric='euclidean', metric='minkowski',
                            random_state=42, tqdm_kwds={'disable': False}).fit_transform(hypEmbedding)
        print('Normalize embedding in Euclidean space ...')
        all_emb = all_emb / max(scipy.linalg.norm(all_emb, axis=1))
        header_ = ['emb_' + str(i) for i in range(np.shape(all_emb)[1])]
        header_ = ['label'] + header_
        df = pd.DataFrame(np.concatenate([np.expand_dims((self.Y).numpy(),axis=1), all_emb],axis=1), columns=header_)
        df.to_csv(all_sample_name,index=False)
        # compute hyp-emb for all unlabeled
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # Run Kmeans to get clusters and sample association
        print('Running Kmean++ in Euclidean space ...')
        chosen = init_centers(all_emb[idxs_unlabeled], n)
        chosen_emb = all_emb[idxs_unlabeled[chosen]]
        header_ = ['emb_' + str(i) for i in range(np.shape(chosen_emb)[1])]
        header_ = ['label', 'index'] + header_
        df = pd.DataFrame(np.concatenate(
            [np.expand_dims((self.Y[chosen]).numpy(), axis=1), np.expand_dims(idxs_unlabeled[chosen], axis=1),
             chosen_emb], axis=1), columns=header_)
        df.to_csv(selected_sample_name, index=False)
        plt.scatter(all_emb.T[0],
                    all_emb.T[1],
                    c=self.Y, cmap='Spectral')
        plt.scatter(chosen_emb.T[0],
                    chosen_emb.T[1],
                    c=self.Y[idxs_unlabeled[chosen]],
                    edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        # emb = np.concatenate(
        #     [np.expand_dims(np.arange(0, len(all_emb)), axis=1), all_emb],
        #     axis=1)[:, 0:3]
        # emb = pd.DataFrame(emb, columns=['node', 'x', 'y'])
        # plot_clusters_no_edge(emb, self.Y, chosen_emb[:,0:2], self.classes)
        plt.savefig(image_name)
        plt.close('all')
        del chosen_emb, all_emb, df
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



