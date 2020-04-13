# imports
import torch
import args_parser
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import datasets, transforms
import torch.nn.functional as F





args = args_parser.parse_arguments()

#cuda enable if available
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')


# Load the checkpoints
def checkpoint_load(name):
    checkpoint = torch.load(name)
    
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    batch_size = checkpoint['batch_size']
    epochs = checkpoint['epochs']
    
    return model, optimizer, checkpoint['class_to_idx']

model, optimizer, class_to_index = checkpoint_load(args.checkpoint)

# Image transformation
def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(img)
    
    transformer = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    transformed = transformer(image)
    
    return transformed

# Image plotting
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.to('cuda')
        
    tensored_image = process_image(image_path)
    
    tensored_image.unsqueeze_(0)
    
    model.eval()
    
    image_input = tensored_image.to('cuda')
    
    # Apply model for feedforward  and to calculate the probabilities
    logps = model.forward(image_input)
    ps = F.softmax(logps, dim = 1)
    
    top_probs, top_cats = ps.topk(topk)
    
    return top_probs, top_cats

#data_dir = args.data_dir
#train_dir = data_dir + '/train'
#image_path = train_dir + '/10/image_07087.jpg'

top_probs, top_cats = predict(args.image, model, args.top_k)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

keys = []
top_cats_list = top_cats.tolist()
for cat in top_cats_list:
    for key, value in class_to_index.items():
        if value in cat:
            keys.append(key)
#print(keys)

probs = top_probs.squeeze().tolist()
print(probs)

species = [cat_to_name[n] for n in keys]
print(species)





