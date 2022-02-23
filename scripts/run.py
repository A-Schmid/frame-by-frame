import warnings
warnings.filterwarnings("ignore")

import config
from VideoHandler import get_processed_videos, VideoHandler
import pandas as pd
from softmax_rdm import create_heatmap, generate_distance_vector
import numpy as np

from hca import hca, create_cophenets_graph, calculate_silhouette, create_silhouette_graph

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
