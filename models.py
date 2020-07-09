## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers 
        # (such as dropout or batch normalization) to avoid overfitting
        
        # channel sizes
        ch1, ch2, ch3, ch4, ch5 = 32, 64, 128, 264, 512
        
        # filter sizes
        fs = [5, 3, 3, 3, 1]
        
        # calculate final max pooling layer dimension
        s = 224 # input dimension
        for f in fs:
            s = (s - f + 1) // 2
        
        # conv layers
        self.conv1 = nn.Conv2d(1, ch1, fs[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(ch1, ch2, fs[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(ch2, ch3, fs[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(ch3, ch4, fs[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(ch4, ch5,fs[4])
        self.pool5 = nn.MaxPool2d(2,2)
        
        # dense layers
        self.fc1 = nn.Linear(s**2 * ch5, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.output = nn.Linear(1000, 136) # predicts (x,y) position of 68 keypoints
        
        # layers to avoid overfitting
        self.norm1 = nn.BatchNorm2d(ch1)
        self.norm2 = nn.BatchNorm2d(ch2)
        self.norm3 = nn.BatchNorm2d(ch3)
        self.norm4 = nn.BatchNorm2d(ch4)
        self.norm5 = nn.BatchNorm2d(ch5)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.6)
        self.drop3 = nn.Dropout(0.6)
     
    
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # conv layers
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = self.norm3(self.pool3(F.relu(self.conv3(x))))
        x = self.norm4(self.pool4(F.relu(self.conv4(x))))
        x = self.norm5(self.pool5(F.relu(self.conv5(x))))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # dense hidden layers
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        
        # output
        x = self.output(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x