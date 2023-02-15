import errno

import numpy as np
import pandas as pd
import os
# plot clustering results in poincare disk
from matplotlib import pyplot as plt

def dist_squared(x, y, axis=None):
    return np.sum((x - y)**2, axis=axis)
def plot_clusters_no_edge(emb, labels, centroids, classes, title=None, height=8, width=8,
                  add_labels=False, label_dict=None, plot_frac=1, label_frac=0.001):
    # Note: parameter 'emb' expects data frame with node ids and coords
    emb.columns = ['node', 'x', 'y']
    n_clusters = len(centroids)
    plt.figure(figsize=(width, height))
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])
    ax = plt.gca()
    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
    ax.add_patch(circ)

    # set colormap
    if n_clusters <= 12:
        colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']
    elif 12 < n_clusters <= 20:
        colors = [i for i in plt.cm.get_cmap('tab20').colors]
    else:
        cmap = plt.cm.get_cmap(name='viridis')
        colors = cmap(np.linspace(0, 1, n_clusters))

    # plot embedding coordinates and centroids
    emb_data = np.array(emb.iloc[:, 1:3])
    for i in range(n_clusters):
        plt.scatter(emb_data[(labels[i] == 1), 0], emb_data[(labels[:, i] == 1), 1],
                    color=colors[i], alpha=0.8, edgecolors='w', linewidth=2, s=250)
        plt.scatter(centroids[i, 0], centroids[i, 1], s=750, color=colors[i],
                    edgecolor='black', linewidth=2, marker='*', label=classes[i])

    ax.legend()
    # add labels to embeddings
    if add_labels and label_dict != None:
        plt.grid('off')
        plt.axis('off')
        embed_vals = np.array(list(label_dict.values()))
        keys = list(label_dict.keys())
        # set threshhold to limit plotting labels too close together
        min_dist_2 = label_frac * max(embed_vals.max(axis=0) - embed_vals.min(axis=0)) ** 2
        labeled_vals = np.array([2*embed_vals.max(axis=0)])
        n = int(plot_frac*len(embed_vals))
        for i in np.random.permutation(len(embed_vals))[:n]:
            if np.min(dist_squared(embed_vals[i], labeled_vals, axis=1)) < min_dist_2:
                continue
            else:
                props = dict(boxstyle='round', lw=2, edgecolor='black', alpha=0.35)
                _ = ax.text(embed_vals[i][0], embed_vals[i][1]+0.02, s=keys[i].split('.')[0],
                            size=10, fontsize=12, verticalalignment='top', bbox=props)
                labeled_vals = np.vstack((labeled_vals, embed_vals[i]))
    if title != None:
        plt.suptitle('Hyperbolic K-Means - ' + title, size=16)
    plt.show()


def create_directory(dir) :

    try:
        # Create inference directory
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(f'Unable to create one of following directories: {dir}')


def save_df_as_npy(path, df):
    """
    Save pandas dataframe (multi-index or non multi-index) as an NPY file
    for later retrieval. It gets a list of input dataframe's index levels,
    column levels and underlying array data and saves it as an NPY file.

    Parameters
    ----------
    path : str
        Path for saving the dataframe.
    df : pandas dataframe
        Input dataframe's index, column and underlying array data are gathered
        in a nested list and saved as an NPY file.
        This is capable of handling multi-index dataframes.

    Returns
    -------
    out : None

    """

    if df.index.nlevels>1:
        lvls = [list(i) for i in df.index.levels]
        lbls = [list(i) for i in df.index.labels]
        indx = [lvls, lbls]
    else:
        indx = list(df.index)

    if df.columns.nlevels>1:
        lvls = [list(i) for i in df.columns.levels]
        lbls = [list(i) for i in df.columns.labels]
        cols = [lvls, lbls]
    else:
        cols = list(df.columns)

    data_flat = df.values.ravel()
    df_all = [indx, cols, data_flat]
    np.save(path, df_all)

def load_df_from_npy(path):
    """
    Load pandas dataframe (multi-index or regular one) from NPY file.

    Parameters
    ----------
    path : str
        Path to the NPY file containing the saved pandas dataframe data.

    Returns
    -------
    df : Pandas dataframe
        Pandas dataframe that's retrieved back saved earlier as an NPY file.

    """

    df_all = np.load(path)
    if isinstance(df_all[0][0], list):
        indx = pd.MultiIndex(levels=df_all[0][0], labels=df_all[0][1])
    else:
        indx = df_all[0]

    if isinstance(df_all[1][0], list):
        cols = pd.MultiIndex(levels=df_all[1][0], labels=df_all[1][1])
    else:
        cols = df_all[1]

    df0 = pd.DataFrame(index=indx, columns=cols)
    df0[:] = df_all[2].reshape(df0.shape)
    return df0

def max_columns(df0, cols=''):
    """
    Get dataframe with best configurations

    Parameters
    ----------
    df0 : pandas dataframe
        Input pandas dataframe, which could be a multi-index or a regular one.
    cols : list, optional
        List of strings that would be used as the column IDs for
        output pandas dataframe.

    Returns
    -------
    df : Pandas dataframe
        Pandas dataframe with best configurations for each row of the input
        dataframe for maximum value, where configurations refer to the column
        IDs of the input dataframe.

    """

    df = df0.reindex_axis(sorted(df0.columns), axis=1)
    if df.columns.nlevels==1:
        idx = df.values.argmax(-1)
        max_vals = df.values[range(len(idx)), idx]
        max_df = pd.DataFrame({'':df.columns[idx], 'Out':max_vals})
        max_df.index = df.index
    else:
        input_args = [list(i) for i in df.columns.levels]
        input_arg_lens = [len(i) for i in input_args]

        shp = [len(list(i)) for i in df.index.levels] + input_arg_lens
        speedups = df.values.reshape(shp)

        idx = speedups.reshape(speedups.shape[:2] + (-1,)).argmax(-1)
        argmax_idx = np.dstack((np.unravel_index(idx, input_arg_lens)))
        best_args = np.array(input_args)[np.arange(argmax_idx.shape[-1]), argmax_idx]

        N = len(input_arg_lens)
        max_df = pd.DataFrame(best_args.reshape(-1,N), index=df.index)
        max_vals = speedups.max(axis=tuple(-np.arange(len(input_arg_lens))-1)).ravel()
        max_df['Out'] = max_vals
    if cols!='':
        max_df.columns = cols
    return max_df
