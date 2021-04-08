# [02.02.21] OV
# Script to change framerate of videos given a parent folder
#%%#############################################################################
# OV 10.02.21
# Make sure to switch to stock (3.8.2) python which has access to homebrew
# installation of the ffmpeg -> let's us encode back into h264 (no visual loss)
################################################################################

#%%#############################################################################
# Imports
################################################################################
import time
from pathlib import Path
import pickle
import os
import sys

################################################################################
#%% Sweep through videos
################################################################################
#%% Paths
#path_input = Path('data/20210301_sampleVideos_additional_fillUpGIFs').absolute()
#path_output = Path('data/MIT_additionalVideos_25FPS').absolute()
path_input = Path('data/single_video').absolute()
path_output = Path('data/single_video_25FPS').absolute()

if not os.path.exists(path_output):
  os.makedirs(path_output)

#%% Sweep through files in subfolders of path_input
l_videos = []
for path, subdirs, files in os.walk(path_input):
  for name in files:
    if name[-3:] == 'mp4':
      l_videos.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

if l_videos:
  l_videos = sorted(l_videos)
print('Total nr. of MP4s: ', len(l_videos))

#%% Sweep through files and change framerate to given i_fps
import subprocess

i_fps = 25
keep_audio = True

start = time.time()

j = 0
i = 0
for category, file_name in l_videos:
  # Verbose
  print(f'{j}/{len(l_videos)}'); j+=1

  path_input_file = str(path_input / category/ file_name)
  path_output_file = str(path_output / category / file_name)
  
  # Create output category directory if not present
  if not os.path.exists(path_output / category):
    os.mkdir(path_output / category)
  
  # Remove file in output dir if present
  if os.path.exists(path_output_file):
    os.remove(path_output_file)
  
  if keep_audio == False: # Do not keep audio
    #l_cmd = ['ffmpeg', '-i', path_input_file, '-r', str(i_fps), '-c:v', 'copy', '-an', '-y', path_output_file]
    l_cmd = ['ffmpeg', '-i', path_input_file, '-c:v', 'libx264', '-r', str(i_fps), '-an', '-y', path_output_file]
    
    out = subprocess.call(l_cmd)
    if out != 0:
      print('Error at ', [category, file_name])
  else: # Keep audio
    #l_cmd = ['ffmpeg', '-i', path_input_file, '-r', str(i_fps), '-c:v', 'copy', '-y', path_output_file]
    l_cmd = ['ffmpeg', '-i', path_input_file, '-c:v', 'libx264', '-r', str(i_fps), '-y', path_output_file]
    
    out = subprocess.call(l_cmd)
    if out != 0:
      print('Error at ', [category, file_name])
    
  # Increment
  i+=1
    
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.2f}s per file)')


# %%
subprocess.call(['ffmpeg', '-i', path_input_file, '-r', str(i_fps), '-c:v', 'copy', '-y', path_output_file])

# %%