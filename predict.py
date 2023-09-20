import torch
from torch import nn
from torchvision import models
import numpy as np
import argparse
from PIL import Image
import json

parser = argparse.ArgumentParser(description='Example with long desc')
parser.add_argument('path_to_image',action='store')
parser.add_argument('checkpoint',action='store')

parser.add_argument('--top_k', action='store', default= 1, type=int)
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true')
arg = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and arg.gpu else 'cpu'

def build_transfer_model(pre_trained_module, in_features, number_of_classes, hidden_units)->nn.Module:
    model = pre_trained_module(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False    
    model.classifier = nn.Sequential(nn.Linear(in_features, hidden_units), # type: ignore
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_units, number_of_classes),
                                    nn.LogSoftmax(dim=1))
    
    return model

def load_checkpoint(checkpoint_filepath) -> nn.Module:
    checkpoint: dict = torch.load(checkpoint_filepath)
    print(checkpoint.keys())
    pre_trained_module = None
    if checkpoint['pre_trained_model_name'] == 'densenet121':
        pre_trained_module = models.densenet121
    model = build_transfer_model(pre_trained_module,
                                 checkpoint['in_features'],
                                 checkpoint['number_of_classes'],
                                 checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image) -> np.ndarray:
    
    with Image.open(image) as img:
        short_side = min(img.size)
        x, y = img.size
        x =int(x * 256/short_side)
        y =int(y * 256/short_side)
        curr_img = img.resize(size=(x, y))
        
        box = ( 0 + (x - 244)/2, 0 + (y - 244)/2,
                x - (x-244)/2, y - (y-244)/2
                )
        curr_img = curr_img.crop(box)
        
        np_img  = np.array(curr_img)/255.0        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])  
        np_img = (np_img - mean) / std
        
        np_img = np_img.transpose((2,0,1))
        return(np_img)

def predict(image_path, checkpoint, topk=arg.top_k) -> tuple[torch.Tensor, torch.Tensor]:
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    img_tensor = torch.from_numpy(process_image(image_path)).unsqueeze(0).type(torch.float).to(device)
    model = load_checkpoint(checkpoint).to(device)
    
    with torch.inference_mode():
        model.eval()
        logps = model(img_tensor)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk,dim=1)
    return probs, classes

probs, classes = predict(arg.path_to_image, arg.checkpoint)

print(f'probs: {probs}')
print(f'classes: {classes}')

with open(arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
print(f'the model classify as: {cat_to_name[str(classes[0].item()+1)]}')

