#Import module to create ModuleL

import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

# Build and train your network
class Network(nn.Module):


    def __init__(self, input_size, output_size, hidden_size, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_size: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()

        '''
            Example from Inference and validations
            # Create ModuleList and add input layer
              hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            # Add hidden layers to the ModuleList
              hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

              hidden_layers = [512, 256, 128, 64]
              layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
              for each in layer_sizes:
                  print(each)

            #drop out is used for more better generalization
            #log_softmax is used in getting away from 0 and 1 and more in -ve numbers(away from inaccuracy)
            #helps computation stable and helps with precision
        '''


        # Add the first layer, input to a hidden_size
        self.hidden_size = nn.ModuleList([nn.Linear(input_size, hidden_size[0])])

        # Add a variable number of more hidden_size
        layer_sizes = zip(hidden_size[:-1], hidden_size[1:])
        self.hidden_size.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_size[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        # Forward through each layer in `hidden_size`, with ReLU activation and dropout
        for linear in self.hidden_size:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)

# validation on the test set

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Define train_NN function
def train_NN(n_hidden, n_epoch, labelsdictionary, lr, device, model_name, trainloader, testloader,  trainset):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # Import pre-trained NN model
    model = getattr(models, model_name)(pretrained=True)

    # Freeze parameters that we don't need to re-train
    for param in model.parameters():
        param.requires_grad = False

    '''
        # Create the network, define the criterion and optimizer
           model = fc_model.Network(784, 10, [512, 256, 128])
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
    '''
    # Make classifier
    n_input = next(model.classifier.modules()).in_features
    n_output = len(labelsdictionary)
    model.classifier = Network(input_size=n_input, output_size=n_output, hidden_size=n_hidden)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

    model.to(device)
    start = time.time()

    epochs = n_epoch
    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion, device)

                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Training Loss: {:.3f} - ".format(running_loss/print_every),
                      "Test Loss: {:.3f} - ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader))))

                running_loss = 0

                # Make sure training is back on
                model.train()

    # Add model info
    model.classifier.n_input = n_input
    model.classifier.n_hidden = n_hidden
    model.classifier.n_output = n_output
    model.classifier.labelsdictionary = labelsdictionary
    model.classifier.lr = lr
    model.classifier.optimizer_state_dict = optimizer.state_dict
    model.classifier.model_name = model_name
    model.classifier.class_to_idx = trainset.class_to_idx

    print('model:', model_name, '- hidden size:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    return model


# Define function to load model
def load_model(path):
    cp = torch.load(path)

    # Import pre-trained NN model
    model = getattr(models, cp['model_name'])(pretrained=True)

    # Freeze parameters that we don't need to re-train
    for param in model.parameters():
        param.requires_grad = False

    # Make classifier
    model.classifier = Network(input_size=cp['checkpoint_input'], output_size=cp['checkpoint_output'], hidden_size=cp['checkpoint_hidden'])

    model.classifier.n_input = cp['checkpoint_input']
    model.classifier.n_hidden = cp['checkpoint_hidden']
    model.classifier.n_output =cp['checkpoint_output']
    model.classifier.labelsdictionary = cp['labelsdictionary']
    model.classifier.lr = cp['checkpoint_state_dict']
    model.classifier.optimizer_state_dict = cp['opti_state_dict']
    model.classifier.model_name = cp['model_name']
    model.classifier.class_to_idx = cp['class_to_idx']

    return model

# Save the checkpoint
def save_the_checkpoint (model, path='checkpoint.pth'):
    print ('start the save of checkpoint')
    checkpoint = {'checkpoint_input': model.classifier.n_input,
                  'checkpoint_hidden': model.classifier.n_hidden,
                  'checkpoint_output': model.classifier.n_output,
                  'labelsdictionary': model.classifier.labelsdictionary,
                  'checkpoint_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'checkpoint_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)
    print ('checkpoint saved as ', path)

# Def model test function

def test_model(model, testloader, device='cpu'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))
