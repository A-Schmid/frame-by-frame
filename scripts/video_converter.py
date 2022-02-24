import sys
import os
import subprocess
import ffmpeg
import config
from pathlib import Path
import pandas as pd

# AS:
# 1. return dataframe with videos?
# 2. support other video formats than mp4?

def get_video_info(video_path):
    # source: Rosario Scavo (CodeHunter) - https://www.codegrepper.com/code-examples/python/python+ffmpeg+get+video+fps
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

    frame_count = int(video_info['nb_frames'].split('/')[0])
    fps = int(video_info['r_frame_rate'])
    duration = float(video_info['duration'])
    width = int(video_info['width'])
    height = int(video_info['height'])
    return {'frame_count' : frame_count,
            'fps' : fps,
            'duration' : duration,
            'width' : width,
            'height' : height}

# deprecated
def get_frame_count(video_path):
    # source: Rosario Scavo (CodeHunter) - https://www.codegrepper.com/code-examples/python/python+ffmpeg+get+video+fps
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    frame_count = int(video_info['nb_frames'].split('/')[0])
    return frame_count

def convert_video(video_path, video_output_path, keep_audio=False, video_fps=25, overwrite=True):
    print(f'converting {video_path}')
    # if not in overwrite mode: skip
    if os.path.exists(video_output_path):
        if overwrite == False:
            return False

    # create output dir if it does not exist
    category_path = video_output_path.rsplit('/', 1)[0]

    if not os.path.exists(category_path):
        os.makedirs(category_path)

    stream = ffmpeg.input(video_path)
    if keep_audio == False:
        stream = stream.video
    stream = stream.filter('fps', fps=video_fps, round='up')
    stream = ffmpeg.output(stream, video_output_path, vcodec='libx264', loglevel='quiet')
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

    return True

def convert_videos(input_path, output_path, data_path=None, keep_audio=False, video_fps=25, overwrite=True):
    video_list = []
    counter = 0
    # maybe dataframe is the better idea?
    for path, subdirs, files in os.walk(input_path):
        for name in files:
            if name[-3:] == 'mp4':
                # support other file types
                category = path.split('/')[-1]
                #file_id = name[:-4].split('_')[-1]
                file_id = counter
                video_path = f'{input_path}/{category}/{name}'
                video_output_path = f'{output_path}/{category}/{name}'
                convert_video(video_path, video_output_path, keep_audio, video_fps, overwrite)
                frame_count = get_frame_count(video_output_path)
                video_list.append({'file_id' : file_id, 'path' : video_output_path, 'num_frames' : frame_count, 'name' : name.rsplit('.', 1)[0], 'category' : category})
                counter += 1
    #df_videos = pd.DataFrame(columns=['file_id', 'path', 'frame_count', 'name', 'category'])
    df_videos = pd.DataFrame(video_list)
    if data_path is not None:
        df_videos.to_csv(f'{data_path}/videos_converted.csv', index=False)
    return df_videos

if __name__ == '__main__':
    if len(sys.argv) > 4:
        df = convert_videos(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        df = convert_videos(input_path=config.VIDEO_PATH, output_path=config.VIDEO_OUTPUT_PATH, data_path=config.DATA_PATH, keep_audio=config.KEEP_AUDIO, video_fps=config.VIDEO_FPS, overwrite=False)
