import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import functional as F
from decord import VideoReader
from decord import cpu, gpu
import decord

from video_converter import convert_videos
import config
#from util import get_video_list

decord.bridge.set_bridge('native')

cuda_on = config.CUDA_ON
model_path = config.MODEL_PATH

print('loading model...')
resnet50 = models.resnet50(pretrained=False, progress=True, num_classes=339)

if cuda_on:
    resnet50 = resnet50.to('cuda')

# Load pretrained weights (MiTv1)
resnet50.load_state_dict(torch.load(model_path))

# AB: source / structure
# Evaluation mode
resnet50.eval()

transformation = transforms.Compose([
                                     transforms.ToPILImage(mode='RGB'), # required if the input image is a nd.array
                                     transforms.Resize(224), # To be changed to rescale to keep the aspect ration?
                                     transforms.CenterCrop((224, 224)), # AB: cropped to 224x224px
                                     transforms.ToTensor(), # AB: creates Tensor (?)
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
])


label_path = config.LABEL_PATH

def load_categories():
    """Load categories."""
    with open(label_path) as f:
        return [line.rstrip() for line in f.readlines()]

print('loading categories...')
# load categories
categories = load_categories()

print('converting videos...')

video_list = convert_videos(input_path=config.VIDEO_PATH, output_path=config.VIDEO_OUTPUT_PATH, keep_audio=config.KEEP_AUDIO, video_fps=config.VIDEO_FPS, overwrite=False)

counter = 0

def calculate_probabilities(video, result_path=config.RESULT_PATH):
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
        img_transformed = transformation(img)
        if cuda_on:
            img_transformed = img_transformed.to_cuda()
        logit = resnet50.forward(img_transformed.unsqueeze(0))
        h_x = F.softmax(logit, 1).data.squeeze()
        probabilities, idx = h_x.sort(0, True)
        for category_index in range(len(probabilities)):
            try:
                if(len(probabilities.data.numpy()[category_index]) > 1):
                    print(probabilities.data.numpy())
            except:
                pass
            df = df.append({'frame' : frame_index, 'action' : categories[idx[category_index]], 'probability' : probabilities.data.numpy()[category_index]}, ignore_index=True)
        df.to_csv(f'{result_path}/{video["category"]}/{video["name"]}_frame_{frame_index:03d}.csv')


print('doing magic...')

for video in video_list:
    print(f'{video["path"]} {counter+1}/{len(video_list)}')
    calculate_probabilities(video, result_path=config.RESULT_PATH)
    counter += 1
