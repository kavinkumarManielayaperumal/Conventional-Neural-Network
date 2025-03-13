import torch 
import torch.nn as nn
import torch.version
import torch.functional as f #this library funcrion without creating in the class , we can use this function directly



class CNN_Model(nn.Model):
    def __init__(self,depth=3,number_of_classes=10):
        super(CNN_Model,self).__init__()
        # here the basic structure of the CNN, and you only need depth of the image and the number of classes
        self.conv1=nn.Conv2d(in_channels=depth,out_channels=16,kernel_size=3,padding=2,stride=1)
        # here the in_channels is 3 becauseis added to the image and stride is 1 which means 1 pixel is moved to the image the image is as RGB images and Out_channels is 16 because we are using 16 filters and kernel size is 3 which means #3x3 filter is used and padding is 2 which means 2 pixel 
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=2,stride=1)
        
        self.max=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=2,stride=1)
        self.conv4=nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3,padding=2,stride=1)
        
        # if you want to do  automate calculation of the image size after the convolutional layer
        #self.fc1=none 
        
        self.fc1=nn.Linear(input_features=128*16*16,hidden_layer=512)
        self.fc2=nn.Linear(input_features=512,output_features=number_of_classes)
        # we can still optimaze the model by using the batch noimalization and dropout layer
        
    def forward(self,x):
        x=f.relu(self.conv1(x))
        x=f.relu(self.conv2(x))
        x=self.max(x)
        x=f.relu(self.conv3(x))
        x=f.relu(self.conv4(x))
        
        #for the automate calculation of the images size after the convolutional layer
        # x=torch.flatten(x,1) # here the 1 is used to flatten the image in the form of 1D array
        #if self.fc1 is None: # if the fc1 is none then the image size is calculated
            #self.fc1=nn.Linear(input_features=x.size(1),hidden_layer=512)
        #x=self.fc1(x)
        
        #we have one more method to calculate
        #x=x.view(x.size(0),-1) # here the -1 is used to calculate the image size automatically
        #x=self.fc1(x)
        
        #but for now we are using the manual calculation of the image size
        x=f.relu(self.fc1(x))
        x=f.relu(self.fc2(x))
        return x
        
        
        