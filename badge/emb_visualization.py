import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from query_strategies.util import create_directory

base_dir = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/'
EXPERIMENT_NAME = 'MNIST_swin_t_UmapHyperboloidKmeans_1000' #os.listdir(results_dir)[3]
results_dir = os.path.join(base_dir, EXPERIMENT_NAME)
demo_file_name = EXPERIMENT_NAME+'_demo_emb.mp4'
lc_file_name = EXPERIMENT_NAME + '_strategy_performance.txt'
input_image_dir = os.path.join(results_dir, 'images')
output_image_dir = os.path.join(results_dir,'images_hyperboloid')
# output_image_dir = os.path.join(results_dir, 'images_poincare')
create_directory(output_image_dir)
output_sample_dir = os.path.join(results_dir,'samples')
learning_accuracy = np.loadtxt(os.path.join(results_dir, lc_file_name))

image_list = sorted(os.listdir(input_image_dir))
[x_min, x_max] = [-1,1]
[y_min, y_max] = [-1,1]
for i in range(len(image_list)):
    selected_emb = np.asarray(pd.read_csv(os.path.join(output_sample_dir, "chosen_{:05d}.csv".format(i))))
    # all_emb = np.asarray((pd.read_csv(os.path.join(output_sample_dir, "all_poincare_{:05d}.csv".format(i)))))
    # all_emb = np.asarray((pd.read_csv(os.path.join(output_sample_dir, "all_euclidean_{:05d}.csv".format(i)))))
    all_emb = np.asarray((pd.read_csv(os.path.join(output_sample_dir, "all_{:05d}.csv".format(i)))))
    chosen_indexes = np.array(selected_emb[:,1], np.dtype(int))
    plt.scatter(all_emb[:,1:].T[0],
                all_emb[:,1:].T[1],
                c=all_emb.T[0], cmap='Spectral')
    plt.scatter(all_emb[chosen_indexes,1:].T[0],
                all_emb[chosen_indexes,1:].T[1],
                c=selected_emb.T[0],
                edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.title('Model accuracy: '+str(round(learning_accuracy[i, 1] * 100, 2))  +'%   '+'Sample size: '+str(int(learning_accuracy[i,0])))
    plt.savefig(os.path.join(output_image_dir, image_list[i]))
    plt.close('all')

# Make demo video
image_list = sorted(os.listdir(output_image_dir))
img_array = []
for filename in image_list:
    img = cv2.imread(os.path.join(output_image_dir, filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(os.path.join(results_dir, demo_file_name), cv2.VideoWriter_fourcc(*'MP4V'), 3, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()