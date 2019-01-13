import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

print ("Import Successful")

model_conv = torchvision.models.inception_v3()

print ("Model created")

print (model_conv)

