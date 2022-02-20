import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import squareform
import numpy as np

metric = 'cosine'

def generate_distance_vector(probability_vector, metric=metric):
    probability_vector = probability_vector.drop(columns=['category'])

    distance_vector = squareform(pdist(np.array(probability_vector), metric=metric))
    return distance_vector

def create_heatmap(distance_vector, categories):

    f, ax = plt.subplots(figsize=(11, 9), num='RDM')

    sns.heatmap(distance_vector, ax=ax, cmap="GnBu", #linewidths=0.01,
                cbar_kws={'label': f'{metric}'},
                square=True,
                xticklabels=categories,
                yticklabels=categories)

    #plt.yticks(fontsize=8)
    #plt.xticks(fontsize=8)
    plt.title('Average Softmax probabilities per MiTv1 category (ResNet50) ')
    plt.show()
