import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image


# load pretrained models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

def main():
    
    # get arguments from command line
    input = get_args()
    
    data_dir = input.data_directory
    save_to = input.save_dir
    pretrained_model = input.arch
    learning_rate = input.learning_rate
    epochs = input.epochs
    hidden_layers = input.hidden_units
    output_size = input.output
    gpu = input.gpu
    drop = 0.5
    
    # load and process images from data directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainloader, validloader, testloader = process(train_dir, valid_dir, test_dir)
    
    # load the pretrained model and determine input size
    model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
    inputsize_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9216}
    
    model = model_dict[pretrained_model]
    input_size = inputsize_dict[pretrained_model]
    
    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # define criterion and optimizer
    classifier = NeuralNetwork(input_size, output_size, hidden_layers, drop)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train model
    print("Training Loss:")
    train(model, trainloader, validloader, criterion, optimizer, epochs, gpu)
    
    # Print Accuracy
    test_loss,test_accuracy = validation(model,testloader,criterion, gpu)
    print("Accuracy (using test data):")
    print("{:.3f} %".format(100*(test_accuracy/len(testloader))) )
    
    # Save Checkpoint
    
    # Save Checkpoint: input, output, hidden layer, epochs, learning rate, model
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'structure': pretrained_model,
                  'learning_rate': learning_rate,
                  'classifier' : model.classifier, 
                  'epochs': epochs,
                  'drop' : drop,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.classifier.state_dict()}

    torch.save(checkpoint, save_to)

# Function Definitions
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_directory", type=str, help="data directory containing training and testing data")
    parser.add_argument("--save_dir", type=str, default="checkpoint_2.pth",
                        help="directory where to save trained model and hyperparameters")
    parser.add_argument("--arch", type=str, default="vgg",
                        help="pre-trained model: vgg16, resnet18, alexnet")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs to train model")
    parser.add_argument("--hidden_units", type=list, default=[700, 300],
                        help="list of hidden layers")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    parser.add_argument("--output", type=int, default=102,
                        help="enter output size")
    
    return parser.parse_args()

def process(train_dir, valid_dir, test_dir):
    """
        Load and transform data so pretrained model can successfully train.
    """
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32,shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True)
    
    return trainloader, validloader, testloader

class NeuralNetwork(nn.Module):
    
    # define layers of the neural network: input, output, hidden layers, dropout
    def __init__(self, input_size, output_size, hidden_layers, drop):
        
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
        self.dropout = nn.Dropout(p = drop)
        
    # feedforward method    
    def forward(self, tensor):
        
        # Feedforward through network using relu activation function
        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
            tensor = self.dropout(tensor)
        tensor = self.output(tensor)
        
        # log_softmax: better for precision (numbers not close to 0, 1)
        return F.log_softmax(tensor, dim=1)



        
# Train function: will use GPU (faster)
def train(model, trainloader, valid_loader, criterion, optimizer, epochs, gpu):
    
    valid_len = len(valid_loader)
    print_every = 30
    #model.train()
    
    steps = 0
    
    # change to cuda
    model.to('cuda')

    for e in range(epochs):
    	model.train()
        running_loss = 0
        val_loss = 0
        for images, labels in trainloader:
            steps += 1
            if gpu==True:
                images, labels = images.to('cuda'), labels.to('cuda')

            # don't want to sum up all gradients for each epoch
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            


            if steps % print_every == 0:
                # inference
                model.eval()
                val_loss, accuracy = validation(model, valid_loader, criterion,gpu)
                
                print("Epoch:{}/{}".format(e+1, epochs), 
                	  "Training Loss:{:.3f}".format(running_loss/print_every),
                      "Validation Loss:{:.3f}".format(val_loss/valid_len),
                      "Validation Accuracy:{:.3f} %".format(100*(accuracy/valid_len)))
                running_loss = 0
                model.train()
            
               
    
# Validation: accuracy and validation set loss
def validation(model, valid_loader, criterion,gpu=True):
    val_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            if gpu==True:
                images, labels = images.to('cuda'), labels.to('cuda')

            #forward
            output = model.forward(images)
            val_loss += criterion(output, labels).item()

            #tensor with probability, tensor with index of flower category
            ps = torch.exp(output) #tensor with prob. of each flower category
            

            #calculate number correct
            equality = (labels.data == ps.max(dim=1)[1])
        	accuracy += equality.type(torch.FloatTensor).mean()
        
    return val_loss, accuracy
        

    
# Run the program
if __name__ == "__main__":
    main()