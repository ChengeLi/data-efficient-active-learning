import os.path
import matplotlib.pyplot as plt
import numpy as np
import umap
import pandas as pd
from query_strategies.util import create_directory
import pdb
import torch

visualize_embedding = True
root = '/workplace/ICCV_AL/data-efficient-active-learning/badge/badge/output/'
# experiment = 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667combinedloss'
# experiment = 'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667_normalized'
# experiment = 'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667' #didn't save emb
# experiment = 'MNIST_HyperNethypNetBadge_500_balldim20_c0.06666666666666667' # #didn't save emb
# experiment = 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667_newnormalized'
# experiment = 'MNIST_HyperNethypNetNorm_500_balldim20_c0.06666666666666667clipr_newlossonly_batchsize250'
# experiment = 'MNIST_HyperNethypNetNorm_500_balldim2_c0.06666666666666667combinedloss'
experiment = 'MNIST_HyperNethypNetBadgePoinKmeans_500_balldim20_c0.06666666666666667clipr'
experiment_path = root+experiment 
original_emb_in_euclidean = False



img_dir = os.path.join(experiment_path,'images3')
emb_dir = os.path.join(experiment_path,'samples')
create_directory(img_dir)
img_path = os.path.join(img_dir, "emb_{:05d}.png".format(98))


for i in range(0, 99, 2):
    img_path = os.path.join(img_dir, "emb_{:05d}.png".format(i))
    # if not os.path.isfile(img_path):
    print("emb_{:05d}".format(i))
    emb = np.load(os.path.join(emb_dir, "emb_{:05d}.npy".format(i)))
    selected_emb = np.asarray(pd.read_csv(os.path.join(emb_dir, "chosen_{:05d}.csv".format(i))))
    chosen_indexes = np.array(selected_emb[:, 1], np.dtype(int))
    Y = emb[:,0]
    emb = emb[:,1:]

    if emb.shape[1]>2:
        if original_emb_in_euclidean:
            pass
        else:
            #project back to euclidean
            from manifolds import PoincareBall
            print('projecting back to euclidean')
            manifold = PoincareBall()
            emb = manifold.logmap0(torch.tensor(emb), c=1/15).numpy()
        hyper_emb = umap.UMAP(output_metric='hyperboloid', random_state=42, tqdm_kwds={'disable': False}).fit_transform(emb)
    else:
        hyper_emb = emb
    x = hyper_emb[:, 0]
    y = hyper_emb[:, 1]
    z = np.sqrt(1 + np.sum(hyper_emb ** 2, axis=1))

    disk_x = x / (1 + z)
    disk_y = y / (1 + z)

    fig = plt.figure() #figsize=(5, 5)
    ax = fig.add_subplot(111)
    ax.scatter(disk_x, disk_y, c=Y, marker='.', cmap='Spectral')
    ax.scatter(disk_x[chosen_indexes],
               disk_y[chosen_indexes],
               c=selected_emb.T[0],
               edgecolor='black', linewidth=1, marker='o', cmap='Spectral')
    boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
    ax.add_artist(boundary)
    # ax.axis('off')
    plt.savefig(img_path)
    plt.close('all')


if visualize_embedding:
    import cv2
    import numpy as np
    video_path = os.path.join(experiment_path,'emb2.mp4')

    if len(os.listdir(img_dir)) > 0:
        img_array = []
        for filename in sorted(os.listdir(img_dir)):
            img = cv2.imread(os.path.join(img_dir, filename))
            img_array.append(img)
        height, width, layers = img_array[-1].shape
        size = (width, height)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), 3, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()