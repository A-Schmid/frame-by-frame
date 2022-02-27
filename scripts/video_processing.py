import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from decord import cpu, gpu
from decord import VideoReader

from video_converter import convert_videos
import config
#from util import get_video_list

from model import calculate_probabilities, load_categories

from time import perf_counter

from VideoHandler import VideoHandler

def process_video(video, result_path=config.RESULT_PATH, process=calculate_probabilities):
    # load categories
    categories = load_categories(config.LABEL_PATH)
    reader = VideoReader(video['path'])
    frames = reader.get_batch([i for i in range(len(reader))])

    frame_data = []

    if os.path.exists(f'{result_path}/{video["category"]}/{video["name"]}.csv'):
        # skip if data already exists
        # Todo: overwrite flag
        return False
    elif not os.path.exists(f'{result_path}/{video["category"]}'):
        os.makedirs(f'{result_path}/{video["category"]}')

    for frame_index in range(len(reader)):

        print(f'frame {frame_index+1}/{len(reader)}')
        img = frames.asnumpy()[frame_index]

        probabilities, idx = process(img)
        for category_index in range(len(probabilities)):
            try:
                if(len(probabilities.data.numpy()[category_index]) > 1):
                    print(probabilities.data.numpy())
            except:
                # more than one probability for a category
                pass
            frame_data.append({'frame' : frame_index, 'action' : categories[idx[category_index]], 'probability' : probabilities.data.numpy()[category_index]})

    #print(frame_data)
    df = pd.DataFrame(frame_data)
    #print(df)
    df.to_csv(f'{result_path}/{video["category"]}/{video["name"]}.csv', index=False)
    return True

def get_processed_videos(path=config.RESULT_PATH):
    categories = os.listdir(path)
    result = []

    for category in categories:
        files = [f for f in os.listdir(f'{path}/{category}')]
        names = []

        for f in files:
            name = str(f).split('.')[0]
            result.append({'name' : name, 'category' : category})

    return result

def create_processed_video_list(path_videos_processed=config.RESULT_PATH, output_path=f'{config.DATA_PATH}/videos_processed.csv'):
    videos = get_processed_videos(path_videos_processed)
    videos_processed_list = []

    for processed_video in videos:
        handler = VideoHandler(processed_video['category'], processed_video['name'])
        video = handler.to_dict()
        videos_processed_list.append(video)

    df_videos_processed = pd.DataFrame(videos_processed_list)
    df_videos_processed.to_csv('{data_path}/videos_processed.csv')

if __name__ == '__main__':

    data_path = config.DATA_PATH

    # todo support args
    path_videos_converted = f'{data_path}/videos_converted.csv'
    path_videos_processed = f'{data_path}/videos_processed.csv'

    df_videos_converted = pd.read_csv(path_videos_converted)
    df_videos_processed = pd.read_csv(path_videos_processed)

    df_videos_converted = df_videos_converted[~df_videos_converted['name'].isin(df_videos_processed['name'])]

    videos_converted_list = df_videos_converted.to_dict('records')

    print('calculating probabilities...')

    counter = 0

    for video in videos_converted_list:
        print(f'{video["path"]} {counter+1}/{len(video_list)}')

        ## Todo: drop videos from video list before even going into the loop
        #if video['name'] in df_videos_processed['name']:
        #    print('already processed - skip...')
        #    continue

        start = perf_counter()
        b = process_video(video, result_path=config.RESULT_PATH, process=calculate_probabilities)
        end = perf_counter()
        if b == True:
            print((end-start))
        counter += 1

    print('creating list of processed videos...')
    create_processed_video_list()
