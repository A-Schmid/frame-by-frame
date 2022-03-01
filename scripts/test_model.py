#%% Imports
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from decord import VideoReader
from random import randint
import matplotlib.pyplot as plt

import config
from base_model import Model

#%% Derived class for a specific model
class ResNet50(Model):
    def __init__(self, labels_path, name='ResNet50', cuda_on=False):
        super().__init__(name)
        self.model = models.resnet50(pretrained=False, progress=True, num_classes=339).eval()
        self.cuda_on = cuda_on
        self.labels = self.load_labels(labels_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize(224),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
            
    def load_pretrained(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        return True
    
    def predict(self, X):
        """Forwards the input Tensor "X" through the network and outputs sorted predicted classes and the indexes pointing to class labels.
        """
        X = self.transform(X)
        if self.cuda_on:
            X = X.to_cuda()
        logit = self.model.forward(X.unsqueeze(0))
        h_x = F.softmax(logit, 1).data.squeeze()
        probabilities, idx = h_x.sort(0, True)
        return probabilities, idx
    
    def predict_test(self, X):
        probabilities, idx = self.predict(X)
        print('--Top Actions:')
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probabilities[i], self.labels[idx[i]]))
            
#%% Create model
my_model = ResNet50(labels_path=config.LABEL_PATH)
categories = my_model.load_labels(config.LABEL_PATH)
my_model.load_pretrained(config.MODEL_PATH)


#%% Load file


# Load video by giving the path to it
# In this case is a video from Kinetics 400
video_fname = '../input_data/test/abseiling_k400.mp4'
vr = VideoReader(video_fname)
print('video frames:', len(vr))

#%% Accuracy test
video_frames = vr.get_batch([i for i in range(len(vr))])
img = video_frames.asnumpy()[randint(0, len(vr))]
plt.imshow(img)
plt.show()

my_model.predict_test(img)
