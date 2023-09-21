import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import json

from helper import build_transfer_model, category_names, setting_up_arch

parser = argparse.ArgumentParser(description='Example with long desc')
parser.add_argument('data_directory',action='store')
parser.add_argument('--save_dir',action='store', default='checkpoint.pth')
parser.add_argument('--arch',action='store',default='densenet121')
parser.add_argument('--hidden_units',action='store',default=512, type=int)
parser.add_argument('--learning_rate',action='store',default=0.003, type=int)
parser.add_argument('--epochs',action='store',default=2, type=int)
parser.add_argument('--gpu', action='store_true')

arg = parser.parse_args()

print(arg)

data_dir = arg.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([
                                      transforms.RandomRotation(30),
                                      transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.RandomHorizontalFlip(0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
                                      
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=32, shuffle=True)
test_loaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=32, shuffle=True)
test_loaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=32, shuffle=True)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    

device = 'cuda' if torch.cuda.is_available() and arg.gpu else 'cpu'

pre_trained_module, in_features = setting_up_arch(arg.arch)
number_of_classes = len(cat_to_name)

model = build_transfer_model(pre_trained_module, in_features, number_of_classes, arg.hidden_units)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arg.learning_rate)
model = model.to(device)

epochs = arg.epochs
step = 0
running_loss = 0
print_every = 10
for epoch in range(epochs):
    for inputs, labels in train_loaders:
        step += 1
       
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        loss = criterion(logps, labels)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        running_loss += loss.item()
       
        if step % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.inference_mode():
                for test_inputs, test_labels in test_loaders:
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_logps  = model(test_inputs)
                    batch_loss = criterion(test_logps, test_labels)
                    
                    valid_loss += batch_loss.item()
                    ps = torch.exp(test_logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == test_labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f'Epoch {epoch +1}/{epochs}'
                 f'Train loss: {running_loss/print_every:.3f}.. '
                 f'valid loss: {valid_loss/len(test_loaders):.3f}.. '
                 f'valid accuracy: {accuracy/len(test_loaders)*100:.2f}% .. ')
            running_loss = 0
            model.train()
           
checkpoint_path = arg.save_dir

checkpoint = {
    "pre_trained_model_name": arg.arch,
    "in_features": in_features,
    "number_of_classes": number_of_classes,
    "hidden_units": arg.hidden_units,
    "state_dict": model.state_dict()
}

torch.save(checkpoint, checkpoint_path)