import warnings
warnings.filterwarnings("ignore")

import config
from VideoHandler import VideoHandler
from video_processing import get_processed_videos
import pandas as pd
from softmax_rdm import create_heatmap, generate_distance_vector
import numpy as np

from hca import hca, create_cophenets_graph, calculate_silhouette, create_silhouette_graph, hierarchical_clustering, create_nested_category_list, create_dendogram, generate_cluster_index, get_cluster_classes

handlers = []

df_probability_vector = pd.read_csv(f'{config.DATA_PATH}/probability_vector.csv')

categories = df_probability_vector['category'].unique()

distance_vector = generate_distance_vector(df_probability_vector)

df_distance = pd.DataFrame(data=distance_vector, columns=categories)
df_distance.to_csv(f'{config.DATA_PATH}/rdm_average_softmax.csv', index=False)

#create_heatmap(distance_vector, categories, save_path=f'{config.IMAGE_PATH}/rdm_matrix.pdf')

cophenets = hca(distance_vector)

#create_cophenets_graph(cophenets, save_path=f'{config.IMAGE_PATH}/cophenets_graph.pdf')

silhouette = calculate_silhouette(distance_vector)

#create_silhouette_graph(silhouette, save_path=f'{config.IMAGE_PATH}/silhouette_graph.pdf')

labels, Z = hierarchical_clustering(distance_vector, silhouette, 'ward')

nested_categories = create_nested_category_list(categories, labels)

df_nested_clusters = pd.DataFrame(nested_categories)
df_nested_clusters.to_csv(f'{config.DATA_PATH}/nested_clusters_{len(labels)}.csv')

R = create_dendogram(categories, Z, save_path=f'{config.IMAGE_PATH}/dendogram.pdf')

cluster_idxs = generate_cluster_index(R)

nested_clusters = get_cluster_classes(cluster_idxs, R)


l_nested_clusters = []

for color in nested_clusters.keys():
    l_nested_clusters.append(nested_clusters[color])

# %% Print items per cluster
for item in nested_clusters.items():
    print(item)
    print('\n')
