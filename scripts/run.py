import config
from VideoHandler import get_processed_videos, VideoHandler
import pandas as pd
from softmax_rdm import create_heatmap, generate_distance_vector
import numpy as np

## HCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics, cluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform

handlers = []

for video in get_processed_videos():
    handlers.append(VideoHandler(video['category'], video['name']))

#df_videos = handlers_to_df(handlers)
#print(df_videos)
#df_videos.to_csv(f'{config.DATA_PATH}/videos.csv')

#print(VideoHandler.video_data)
#print(VideoHandler.handlers)
#print(VideoHandler.get_accuracy_for_category('arresting'))
#print(VideoHandler.get_probability_vector_for_category('arresting'))

probability_vector = []

categories = VideoHandler.get_categories()
for category in categories:
    row = VideoHandler.get_probability_vector_for_category(category)
    probability_vector.append(row)

df_probability_vector = pd.DataFrame(probability_vector)
df_probability_vector.set_index('category')
df_probability_vector = df_probability_vector.sort_values(by=['category'])

#print(df_probability_vector[['category', 'arresting', 'attacking', 'brushing', 'building', 'buying']])
#print(np.array(df_probability_vector[['category', 'arresting', 'attacking', 'brushing', 'building', 'buying']]))

distance_vector = generate_distance_vector(df_probability_vector)
create_heatmap(distance_vector, categories)

df_distance = pd.DataFrame(data=distance_vector, columns=categories)
df_distance.to_csv(f'{config.DATA_PATH}/rdm_average_softmax.csv', index=False)


## HCA

# can't proceed as I don't have utils

#linkage_types = ['ward', 'complete', 'average', 'single']
#l_cophenets = []
#
#for linkage in linkage_types:
#    # Fit model on the selected linkage
#    model = cluster.AgglomerativeClustering(linkage=linkage)
#    model.fit(distance_vector)
#    
#    # Compute distances between clusters
#    distances, weights = utils.get_distances(distance_vector, model, 'max')
#    Z = np.column_stack([model.children_, distances, weights]).astype(float)
#    
#    # Compute cophenetic correlation coeff
#    c, d = cophenet(Z, squareform(distance_vector))
#    l_cophenets.append([c, linkage])
#print(l_cophenets)
