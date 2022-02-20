import config
from VideoHandler import get_processed_videos, VideoHandler
import pandas as pd
from softmax_rdm import create_heatmap, generate_distance_vector
import numpy as np

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

for category in VideoHandler.get_categories():
    row = VideoHandler.get_probability_vector_for_category(category)
    probability_vector.append(row)

df_probability_vector = pd.DataFrame(probability_vector)
df_probability_vector.set_index('category')
df_probability_vector = df_probability_vector.sort_values(by=['category'])

#print(df_probability_vector[['category', 'arresting', 'attacking', 'brushing', 'building', 'buying']])
#print(np.array(df_probability_vector[['category', 'arresting', 'attacking', 'brushing', 'building', 'buying']]))

distance_vector = generate_distance_vector(df_probability_vector)
create_heatmap(distance_vector, VideoHandler.get_categories())
