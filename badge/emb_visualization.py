import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
results_dir = '/home/ubuntu/workplace/code/data-efficient-active-learning/badge/output/'
EXPERIMENT_NAME = 'swin_t_hypUmap'
learning_accuracy = np.loadtxt(os.path.join(results_dir, EXPERIMENT_NAME + '_strategy_performance.txt'))
all_emb = pd.read_csv()
plt.scatter(all_euclidean_emb.T[0],
            all_euclidean_emb.T[1],
            c=all_euclidean_emb.T[2], cmap='Spectral')
plt.scatter(chosen_euclidean_emb.T[0],
            chosen_euclidean_emb.T[1],
            c=chosen_euclidean_emb.T[2],
            edgecolor='black', linewidth=0.3, marker='*', cmap='Spectral')
plt.xlim([-1, 1])
plt.ylim([-1, 1])