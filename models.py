## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def conv(in_f,out_f,ker,pad,dp):
    
    layers = []
    layers.append(nn.Conv2d(in_f,out_f, ker,padding = pad))
    layers.append(nn.BatchNorm2d(out_f))
    layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.MaxPool2d(2,2))
    layers.append(nn.Dropout2d(dp))
                  
    layers_seq = nn.Sequential(*layers)
    return layers_seq
    

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = conv(1, 32, 5, 2, 0.3)
        self.conv2 = conv(32, 64 ,5, 2, 0.4)
        self.conv3 = conv(64, 128, 3, 1, 0.5)
        self.conv4 = conv(128, 256, 3, 1, 0.6)
        self.conv5 = conv(256, 512, 3, 1, 0.6)
        
        self.fc1 = nn.Linear(512*7*7,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3= nn.Linear(512,136)
         
        self.dropout = nn.Dropout2d(0.4)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0),-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = (self.fc3(x))
     
        # a modified x, having gone through all the layers of your model, should be returned
        return x
