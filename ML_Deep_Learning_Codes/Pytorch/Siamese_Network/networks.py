import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Defining CNN architecture                                         
                                                                    
class Net(nn.Module):                                               
    def __init__(self):                                             
        super(Net, self).__init__()                                 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)                
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)               
        self.conv2_drop = nn.Dropout2d()                            
        self.fc1 = nn.Linear(320, 50)                               
        self.fc2 = nn.Linear(50, 10)                                
                                                                    
    def forward(self, x):                                           
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                  
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  
        x = x.view(-1, 320)                                         
        x = F.relu(self.fc1(x))                                     
        x = F.dropout(x, training=self.training)                    
        x = self.fc2(x)                                             
        return x                                                    
                                                                    
class SomeNet(nn.Module):                                           
                                                                    
    def __init__(self):                                             
        super(SomeNet, self).__init__()                             
        self.convnet = torchvision.models.alexnet(pretrained=False) 
        self.bottleneck = nn.Sequential(nn.Linear(1000, 256))           
        #self.bottleneck = nn.Linear(1000, 256)                     
                                                                    
    def forward(self, x):                                           
        x = self.convnet(x)                                         
        x = self.bottleneck(x)                                      
        return x 
