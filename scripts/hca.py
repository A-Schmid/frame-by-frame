import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics, cluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform

import utils

linkage_types = ['ward', 'complete', 'average', 'single']

def hca(distance_vector):
    cophenets = dict()

    for linkage in linkage_types:
        # Fit model on the selected linkage
        model = cluster.AgglomerativeClustering(linkage=linkage)
        model.fit(distance_vector)
        
        # Compute distances between clusters
        distances, weights = utils.get_distances(distance_vector, model, 'max')
        Z = np.column_stack([model.children_, distances, weights]).astype(float)
        
        # Compute cophenetic correlation coeff
        c, d = cophenet(Z, squareform(distance_vector))
        cophenets[linkage] = c

    return cophenets

def create_cophenets_graph(cophenets, save_path=None):
    df = pd.DataFrame.from_records([cophenets])

    fig, axes = plt.subplots(num='Cophenetic')

    sns.barplot(data=df, ax=axes)
    plt.title('Cophenetic correlation coefficients for different HC linkages')
    #plt.show()
    if save_path is not None:
        plt.savefig(save_path, layout='thight')


def calculate_silhouette(features, linkage='ward', metric='precomputed'):
    silhouettes = []

    # Todo: magic numbers
    for N in range(50,260):
        model = cluster.AgglomerativeClustering(n_clusters=N,
                                                linkage=linkage)
        model.fit(features)
        silhouette_coef = metrics.silhouette_score(features,
                                                   labels=model.labels_,
                                                   metric=metric)
        silhouettes.append({'N' : N, 'coefficient' : silhouette_coef})

    return silhouettes


def create_silhouette_graph(silhouettes, linkage='ward', save_path=None):
    df = pd.DataFrame(silhouettes)

    fig, axes = plt.subplots(num='Silhouette\'s Idx')

    sns.lineplot(data=df, x='N', y='coefficient', marker='o', ax=axes)

    max_coefficient = df['coefficient'].max()
    for peak in df[df['coefficient'] == max_coefficient]['N'].values:
        # Todo: round value?
        plt.axvline(peak, color='red', label=f'max(Silhouette) = {max_coefficient}')

    plt.legend()
    plt.title('Silhouette\'s Idx for {linkage} linkage')
    #plt.show()
    if save_path is not None:
        plt.savefig(save_path, layout='thight')
