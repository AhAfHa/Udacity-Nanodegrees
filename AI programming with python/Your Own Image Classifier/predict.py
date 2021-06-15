import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

def main():
    
    # get arguments from command line
    input = get_args()
    
    path_to_image = input.image_path
    checkpt = input.checkpoint
    num = input.top_k
    cat_names = input.category_names
    gpu = input.gpu
    
    # load category names file
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # load trained model
    model = load(checkpt)
    
    # Process images, predict classes, and display results
    img = Image.open(path_to_image)
    image = process_image(img)
    probs, classes = predict(path_to_image, model, num)
    check(path_to_image, model)
    
    
    

# Function Definitions
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    
    return parser.parse_args()

# define NeuralNetwork Class with FeedForward Method
class NeuralNetwork(nn.Module):
    # define layers of the neural network: input, output, hidden layers
    def __init__(self, input_size, output_size, hidden_layers):

        # calls init method of nn.Module (base class)
        super().__init__()

        # input_size to hidden_layer 1 : ModuleList --> list meant to store nn.Modules
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # add arbitrary number of hidden layers
        i = 0
        j = len(hidden_layers)-1

        while i != j:
            l = [hidden_layers[i], hidden_layers[i+1]]
            self.hidden_layers.append(nn.Linear(l[0], l[1]))
            i+=1

        # check to make sure hidden layers formatted correctly
        for each in hidden_layers:
            print(each)

        # last hidden layter -> output
        self.output = nn.Linear(hidden_layers[j], output_size)

    # feedforward method    
    def forward(self, tensor):

        # Feedforward through network using relu activation function
        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
        tensor = self.output(tensor)

        # log_softmax: better for precision (numbers not close to 0, 1)
        return F.log_softmax(tensor, dim=1)
    
    
def load_checkpoint(filepath):
    """
        Load the saved trained model inorder to use for prediction
    """
    checkpoint = torch.load(filepath)
    model = getattr(torchvision.models, checkpoint['structure'])(pretrained=True)
    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = NeuralNetwork(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'],
                             checkpoint['drop'])
    model.classifier = classifier

    
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    model.classifier.optimizer = checkpoint['optimizer']
    model.classifier.epochs = checkpoint['epochs']
    model.classifier.learning_rate = checkpoint['learning_rate']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    img = img_transform(pil_image)
    
    return img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    with torch.no_grad():
        output = model.forward(img.cuda())
    ps = F.softmax(output.data,dim=1)
    return ps.topk(topk)
    

def check(image_path, model):
	"""
        Ouput a picture of the image and a graph representing its top 'k' class labels
    """
    fig,ax = plt.subplots()
    img = process_image(image_path)
    probs,classes = predict(image_path,model)
    prob = np.array(probs[0])
    names = [cat_to_name[str(index + 1)] for index in np.array(classes[0])]
    imshow(img)
    ax.barh(names,prob,align ='center')
    ax.set_yticklabels(names)
    ax.set_ylabel("Predicted Flower Species")
    ax.set_xlabel("Probabilities")
    
    plt.show()

    
# Run the program
if __name__ == "__main__":
    main()