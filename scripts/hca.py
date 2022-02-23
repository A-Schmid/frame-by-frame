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
    silhouette = []

    # Todo: magic numbers
    for N in range(50,260):
        model = cluster.AgglomerativeClustering(n_clusters=N,
                                                linkage=linkage)
        model.fit(features)
        silhouette_coef = metrics.silhouette_score(features,
                                                   labels=model.labels_,
                                                   metric=metric)
        silhouette.append({'N' : N, 'coefficient' : silhouette_coef})

    return silhouette


def create_silhouette_graph(silhouette, linkage='ward', save_path=None):
    df = pd.DataFrame(silhouette)

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


def hierarchical_clustering(features, silhouette, linkage):
    # Todo: there has to be a nicer way to do this
    # maybe just use a data frame?
    max_coefficient = silhouette[0]['coefficient']
    max_coefficient_index = 0
    for i in range(len(silhouette)):
        if silhouette[i]['coefficient'] > max_coefficient:
            max_coefficient = silhouette[i]['coefficient']
            max_coefficient_index = i

    model = cluster.AgglomerativeClustering(n_clusters=max_coefficient_index,
                                            #affinity='precomputed',
                                            linkage=linkage, #'ward', #'complete',
                                            distance_threshold=None)
    model.fit(features)

    distances, weights = utils.get_distances(features, model, 'max')

    # Todo: what is Z?
    Z = np.column_stack([model.children_, distances, weights]).astype(float)

    return model.labels_, Z

def create_nested_category_list(categories, labels):
    l_nested_categories = []
    categories = np.array(categories)

    for i in range(max(labels)):
        cat = categories[labels == i]
        x = []
        for label in cat:
            x.append(label)
            
        l_nested_categories.append(x)

    return l_nested_categories

def create_dendogram(labels, Z, save_path=None):
    plt.figure(figsize=(5,20), num='Dendrogram') # AB:Create visualization
    R = dendrogram(
                    Z,
                    orientation='left',
                    labels=labels,
                    #truncate_mode='level',
                    #p=38,
                    distance_sort='descending',
                    show_leaf_counts=False,
                    #leaf_rotation=45,
                    #color_threshold=1.1, # 1.7
                    above_threshold_color='lightgray',
                    show_contracted=True,
                    #leaf_font_size=3.,
                    #link_color_func=lambda k: colors[k]
              )
    #plt.show()
    if save_path is not None:
        plt.savefig(save_path, layout='thight')
