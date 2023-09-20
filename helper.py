import torch
import numpy as np
from torch import nn
from torchvision import models
from PIL import Image

category_names = 'cat_to_name.json'

def build_transfer_model(pre_trained_module_Callable, in_features, number_of_classes, hidden_units)->nn.Module:
    model = pre_trained_module_Callable(pretrained=True)
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
    pre_trained_module = None
    if checkpoint['pre_trained_model_name'] == 'densenet121':
        pre_trained_module = models.densenet121
    model = build_transfer_model(pre_trained_module,
                                 checkpoint['in_features'],
                                 checkpoint['number_of_classes'],
                                 checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path) -> np.ndarray:
    
    with Image.open(image_path) as img:
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
    
def predict(image_path, checkpoint_path, topk, device) -> tuple[torch.Tensor, torch.Tensor]:
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    img_tensor = torch.from_numpy(process_image(image_path)).unsqueeze(0).type(torch.float).to(device)
    model = load_checkpoint(checkpoint_path).to(device)
    
    with torch.inference_mode():
        model.eval()
        logps = model(img_tensor)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk,dim=1)
    return probs, classes