# imports
import args_parser
import data_process

import pandas as pd
import numpy as np

import torch
import torchvision.models as models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
#from workspace_utils import active_session
from PIL import Image


args = args_parser.parse_arguments()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

dataloader_train, dataloader_valid, dataloader_test, train_datasets = data_process.data_loader(train_dir, valid_dir, test_dir, args.batch_size)

if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
    
# define classifier for the network
input_size    = 25088          #input size for vgg16 is 224*224/2
output_size   = 102            #number of different flower categories
hidden_layers = args.hidden_units

classifier = nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(0.5)),
                    ('fc_1', nn.Linear(input_size, hidden_layers)),
                    ('relu_1', nn.ReLU()),
                    ('fc_2', nn.Linear(hidden_layers, output_size)),
                    ('output', nn.LogSoftmax(dim=1))
                    ]))

for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier
print(model)

#cuda enable if available
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
# Parameters
epochs = args.epochs
steps = 0
learning_rate = args.learning_rate
print_every = 10

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Validation
def do_validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    
    model.eval()
    
    for images, labels in iter(testloader):
        
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equals = (labels.data == ps.max(dim=1)[1])
        accuracy += equals.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

model.to(device)

# Training
for e in range(args.epochs):
    running_loss = 0
    
    model.train()
    
    for images, labels in iter(dataloader_train):
        steps += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            
            with torch.no_grad():
                validation_loss, accuracy = do_validation(model, dataloader_valid, criterion)
        
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(validation_loss/len(dataloader_valid)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(dataloader_test)))


# Validation on the test set
model.eval()
    
with torch.no_grad():
    losses, accuracy = do_validation(model, dataloader_test, criterion)
                
print("Test Accuracy: {:.2f}%".format(accuracy*100/len(dataloader_test)))  

            
print(args)

# Saving checkpoint 
model.class_to_idx = train_datasets.class_to_idx

checkpoint = {'model': model,
              'input_size' : input_size,
              'hidden_layers': args.hidden_units,
              'output_size' : output_size,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'epochs': args.epochs,
              'optimizer': optimizer,
              'batch_size': args.batch_size
             }
            
torch.save(checkpoint, args.saved_checks)




