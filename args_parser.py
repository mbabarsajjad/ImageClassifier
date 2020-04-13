# imports
from argparse import ArgumentParser


# Arguments parsing function
def parse_arguments():
    
    parser = ArgumentParser(description = 'Image Classifier checkpoints parser')

    # checkpoints for command line application
    parser.add_argument('--arch', 
                        action = 'store', 
                        default  = 'vgg16', 
                        type = str, help = 'please specify the model: vgg16 or vgg19')
    parser.add_argument('--hidden_units', 
                        action = 'store', 
                        default  = '4096', 
                        type = int, help = 'please specify the hidden units')
    parser.add_argument('--learning_rate', 
                        action = 'store', 
                        default  = '0.001', 
                        type = float, help = 'please specify the learning rate')
    parser.add_argument('--epochs', 
                        action = 'store', 
                        default  = '4', 
                        type = int, help = 'please specify the epochs')
    parser.add_argument('--batch_size', 
                        action = 'store', 
                        default  = '64', 
                        type = int, help = 'please specify the batch size')
    parser.add_argument('--data_dir', 
                        action = 'store', 
                        default  = 'flowers', 
                        type = str, help = 'please specify the path to the flowers data')
    parser.add_argument('--gpu', 
                        action = 'store_true', 
                        help = 'please enable/disable the gpu mode')
    parser.add_argument('--saved_checks', 
                        action = 'store', 
                        default  = 'saved_checkpoints', 
                        type = str, help = 'please specify the path for checkpoints')
    parser.add_argument('--top_k', 
                        action = 'store',
                        default = 5,
                        type = int, help =' Please specify the values for top_k')
    parser.add_argument('--category_names', 
                        action = 'store', 
                        default  = 'cat_to_name.json', 
                        type = str, help = 'please specify the file for category names')
    parser.add_argument('--image', 
                        type = str, 
                        help = 'Please specify path/to/image as an input',
                        required = True)
    parser.add_argument('--checkpoint', 
                        type = str, 
                        help = 'Please specify the checkpoint file',
                        required = True)
       
    # add arguments to return
    args = parser.parse_args()
    
    return args
