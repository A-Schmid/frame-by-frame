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

    fig, axes = plt.subplots()

    sns.barplot(data=df, ax=axes)
    #plt.show()
    if save_path is not None:
        plt.savefig(save_path, layout='thight')
