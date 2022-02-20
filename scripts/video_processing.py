import os
import pandas as pd
from decord import cpu, gpu
from decord import VideoReader

from video_converter import convert_videos
import config
#from util import get_video_list

from model import calculate_probabilities, load_categories

def process_video(video, result_path=config.RESULT_PATH, process=calculate_probabilities):
    reader = VideoReader(video['path'])
    frames = reader.get_batch([i for i in range(len(reader))])

    for frame_index in range(len(reader)):
        if os.path.exists(f'{result_path}/{video["category"]}/{video["name"]}_frame_{frame_index:03d}.csv'):
            # skip if data already exists
            # Todo: overwrite flag
            continue
        elif not os.path.exists(f'{result_path}/{video["category"]}'):
            os.makedirs(f'{result_path}/{video["category"]}')

        df = pd.DataFrame(columns=['frame', 'action', 'probability'])

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
            df = df.append({'frame' : frame_index, 'action' : categories[idx[category_index]], 'probability' : probabilities.data.numpy()[category_index]}, ignore_index=True)
        df.to_csv(f'{result_path}/{video["category"]}/{video["name"]}_frame_{frame_index:03d}.csv', index=False)


if __name__ == '__main__':
    print('loading categories...')
    # load categories
    categories = load_categories(config.LABEL_PATH)

    print('converting videos...')

    video_list = convert_videos(input_path=config.VIDEO_PATH, output_path=config.VIDEO_OUTPUT_PATH, keep_audio=config.KEEP_AUDIO, video_fps=config.VIDEO_FPS, overwrite=False)

    print('doing magic...')

    counter = 0

    for video in video_list:
        print(f'{video["path"]} {counter+1}/{len(video_list)}')
        process_video(video, result_path=config.RESULT_PATH, process=calculate_probabilities)
        counter += 1
