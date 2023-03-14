import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from query_strategies.util import create_directory

visualize_embedding = True
experiment_path = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/CIFAR10_net00_embDim20_c1-15_PoincareKmeans_500'
# experiment_path = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/CIFAR10_net00_embDim20_c1-15_meal_500'
# experiment_path = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/CIFAR10_net00_embDim20_c1-15_badge_500'
# experiment_path = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/MNIST_net00_embDim20_c1-15_PoincareKmeansUncertainty_500'
# experiment_path = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/MNIST_net00_embDim20_meal_500'
# experiment_path = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/MNIST_net00_embDim20_badge_500/'
# experiment_path = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/MNIST_net00_embDim20_curvature-1-15_PoincareKmeans_500/'
img_dir = os.path.join(experiment_path,'images3')
emb_dir = os.path.join(experiment_path,'samples')
create_directory(img_dir)
img_path = os.path.join(img_dir, "emb_{:05d}.png".format(98))

for i in range(0, 99, 2):
    img_path = os.path.join(img_dir, "emb_{:05d}.png".format(i))
    if not os.path.isfile(img_path):
        print("emb_{:05d}".format(i))
        emb = np.load(os.path.join(emb_dir, "emb_{:05d}.npy".format(i)))
        selected_emb = np.asarray(pd.read_csv(os.path.join(emb_dir, "chosen_{:05d}.csv".format(i))))
        chosen_indexes = np.array(selected_emb[:, 1], np.dtype(int))
        Y = emb[:,0]
        emb = emb[:,1:]

        hyper_emb = umap.UMAP(output_metric='hyperboloid',n_epochs=1, random_state=42, tqdm_kwds={'disable': False}).fit_transform(emb)

        x = hyper_emb[:, 0]
        y = hyper_emb[:, 1]
        z = np.sqrt(1 + np.sum(hyper_emb ** 2, axis=1))

        disk_x = x / (1 + z)
        disk_y = y / (1 + z)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.scatter(disk_x, disk_y, c=Y, marker='.', cmap='Spectral')
        scatter = ax.scatter(disk_x[chosen_indexes],
                   disk_y[chosen_indexes],
                   c=selected_emb.T[0],
                   edgecolor='black', linewidth=1, marker='o', cmap='Spectral')
        boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
        ax.add_artist(boundary)
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Classes")
        ax.add_artist(legend1)
        ax.axis('off')
        # plt.savefig(img_path)
        plt.close('all')


if visualize_embedding:
    import cv2
    import numpy as np
    video_path = os.path.join(experiment_path,'emb.mp4')

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