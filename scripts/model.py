import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import decord

import config

def load_categories(label_path):
    """Load categories."""
    with open(label_path) as f:
        return [line.rstrip() for line in f.readlines()]

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

def calculate_probabilities(img, model=resnet50):
    img_transformed = transformation(img)
    if cuda_on:
        img_transformed = img_transformed.to_cuda()
    logit = model.forward(img_transformed.unsqueeze(0))
    h_x = F.softmax(logit, 1).data.squeeze()
    probabilities, idx = h_x.sort(0, True)
    return probabilities, idx
