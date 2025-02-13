import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


CLASSES_NUMBER = 6


def model_fn(model_dir):
    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, CLASSES_NUMBER))

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def input_fn(request_body, content_type):
    image = Image.open(io.BytesIO(request_body))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), normalize])

    return transformation(image).unsqueeze(0)


