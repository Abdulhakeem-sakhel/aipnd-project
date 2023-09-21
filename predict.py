import torch
from torch import nn
from torchvision import models
import numpy as np
import argparse
from PIL import Image
import json

from helper import predict, category_names

parser = argparse.ArgumentParser(description='Example with long desc')
parser.add_argument('path_to_image',action='store')
parser.add_argument('checkpoint',action='store')

parser.add_argument('--top_k', action='store', default= 1, type=int)
parser.add_argument('--category_names', action='store', default=category_names)
parser.add_argument('--gpu', action='store_true')
arg = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and arg.gpu else 'cpu'



probs, classes = predict(arg.path_to_image, arg.checkpoint,arg.top_k, device)
probs, classes = probs.squeeze(), classes.squeeze()

print(f'probs: {probs}')
print(f'classes: {classes}')

with open(arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
print(f'the model classify the image as: {cat_to_name[str(classes[0].item()+1)]}')

