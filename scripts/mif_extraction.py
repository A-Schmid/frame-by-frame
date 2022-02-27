import config
from VideoHandler import VideoHandler
import pandas as pd
from video_converter import get_frame_from_video
import os

output_path = f'{config.MIF_PATH}'

df = pd.read_csv(f'{config.DATA_PATH}/videos_processed.csv')

data_list = df.to_dict('records')

counter = 0

for row in data_list:
    print(f'{counter}/{len(df.index)}')
    category = row['category']
    name = row['name'].split('.')[0]
    mif = row['mif_index']
    path = row['path']

    if not os.path.exists(f'{output_path}/{category}'):
        os.makedirs(f'{output_path}/{category}')

    get_frame_from_video(path, f'{output_path}/{category}/{name}.jpg', mif)
    counter += 1
