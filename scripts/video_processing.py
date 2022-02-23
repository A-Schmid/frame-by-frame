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

def process_video(video, result_path=config.RESULT_PATH, process=calculate_probabilities):
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

if __name__ == '__main__':
    print('loading categories...')
    # load categories
    categories = load_categories(config.LABEL_PATH)

    data_path = config.DATA_PATH

    # todo support args
    path_videos_converted = f'{data_path}/videos_converted.csv'

    #video_list = convert_videos(input_path=config.VIDEO_PATH, output_path=config.VIDEO_OUTPUT_PATH, keep_audio=config.KEEP_AUDIO, video_fps=config.VIDEO_FPS, overwrite=False)

    df_videos = pd.read_csv(path_videos_converted)
    video_list = df_videos.to_dict('records')

    print('doing magic...')

    counter = 0

    for video in video_list:
        print(f'{video["path"]} {counter+1}/{len(video_list)}')
        start = perf_counter()
        b = process_video(video, result_path=config.RESULT_PATH, process=calculate_probabilities)
        end = perf_counter()
        if b == True:
            print((end-start))
        counter += 1
