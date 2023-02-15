import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


results_dir = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/'
EXPERIMENT_NAME = os.listdir(results_dir)[3]
output_image_dir = os.path.join(results_dir,EXPERIMENT_NAME,'images')
output_sample_dir = os.path.join(results_dir,EXPERIMENT_NAME,'samples')
learning_accuracy = np.loadtxt(os.path.join(results_dir,EXPERIMENT_NAME, EXPERIMENT_NAME + '_strategy_performance.txt'))

image_list = sorted(os.listdir(output_image_dir))
[x_min, x_max] = [-0.2,0.8]
[y_min, y_max] = [-0.2,0.8]
for i in range(len(image_list)):
    selected_emb = np.asarray(pd.read_csv(os.path.join(output_sample_dir, "chosen_{:05d}.csv".format(i))))
    all_emb = np.asarray((pd.read_csv(os.path.join(output_sample_dir, "all_{:05d}.csv".format(i)))))

    plt.scatter(all_emb[:,1:].T[0],
                all_emb[:,1:].T[1],
                c=all_emb.T[0], cmap='Spectral')
    plt.scatter(selected_emb[:,2:].T[0],
                selected_emb[:,2:].T[1],
                c=selected_emb.T[0],
                edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.title('Model accuracy: '+str(round(learning_accuracy[i, 1] * 100, 2))  +'%   '+'Sample size: '+str(int(learning_accuracy[i,0])))
    plt.savefig(os.path.join(output_image_dir,image_list[i]))
    plt.close('all')

# Make demo video
img_array = []
for filename in image_list:
    img = cv2.imread(os.path.join(output_image_dir, filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(os.path.join(results_dir,EXPERIMENT_NAME, EXPERIMENT_NAME+'_demo_emb.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 3, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()