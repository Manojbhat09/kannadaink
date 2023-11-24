# usage: python inference.py --model_path path_to_your_model --image_path path_to_your_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import os

class CharacterClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def classify(self, image):
        return classify_image(self.model, image)
    

# -----------------------------------  self densenet start -------------------------------------- #    
class DenseBlock3(torch.nn.Module):
    def __init__(self,in_dim,channels):
        super(DenseBlock3,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim,channels,3,1,1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        self.conv2 = nn.Sequential(nn.Conv2d(channels,channels,3,1,1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        self.conv3 = nn.Sequential(nn.Conv2d(channels,channels,3,1,1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(channels))
        # pass self init
    def forward(self,input_):
        x0 = self.conv1(input_)
        x1 = self.conv2(x0)
        x2 = self.conv3(x1)
        x_out = torch.cat((x0,x1,x2),1)
        return x_out
    
class DenseBlock2(torch.nn.Module):
    def __init__(self,in_dim,channels):
        super(DenseBlock2,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim,channels,3,1,1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(channels))
        self.conv2 = nn.Sequential(nn.Conv2d(channels,channels,3,1,1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(channels))
        # pass self init
    def forward(self,input_):
        x0 = self.conv1(input_)
        x1 = self.conv2(x0)
        x_out = torch.cat((x0,x1),1)
        return x_out

class SelfDenseNet(torch.nn.Module):
    '''
    I didn't rewrite the init portion of this network
    '''
    def __init__(self,args):
        super(SelfDenseNet,self).__init__()
        self.net_cnn = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,1,1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            
            DenseBlock2(32,32),
            nn.Dropout(0.4),
            DenseBlock2(64,32),
            nn.MaxPool2d(2,2),
            
            DenseBlock3(64,64),
            DenseBlock3(64*3,64),
            nn.Dropout(0.4),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64*3,128,3,1,0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.4),   
        )
        self.fc1 = nn.Linear(128,10)
        
    def forward(self,input_):
        x = self.net_cnn(input_)       
        x = x.view(x.size(0),-1)
        x = torch.nn.functional.softmax(self.fc1(x),dim=-1)     
        return x

# -----------------------------------  self densenet end -------------------------------------- # 


ROOT ='/home/mbhat/'
DATASET =os.path.join(ROOT, 'ocr/recognition-model/dataset')
REPO_PATH = os.path.join(ROOT, 'ocr/recognition-model')

class argsModel(object):
    # ----------------------------------- network ----------------------------------- #
    opt = 'rms'
    batch_size = 1024#256#16
    lr = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0#1e-4
    epoch = 56#40#62
    factor= 0.25#0.1
    step = [2000,2500,2700]
    patience = 3
    auto_lr_type = 'auto'
    # -----------------------------------  data ------------------------------------ #
    train_path = os.path.join(DATASET, 'train.csv')
    val_path = os.path.join(DATASET, 'Dig-MNIST.csv')
    test_path = os.path.join(DATASET, 'test.csv')
    # result_path = './submission2.csv'
    weight_path = os.path.join(REPO_PATH, 'best.pth')
    weight_save_path = os.path.join(REPO_PATH, 'saved.pth')
    aug_train = True
    crop_padding = 3
    scale = (0.60,1.40)#(0.75,1.25)
    shear = 0.15#0.10
    shift = (0.15,0.15)#(0.25,0.25)
    angle = (-15,15)#(-10,10)
    n_splits = 5
    n_times = 3 
    # -----------------------------------  other ------------------------------------ #
    SEED = 0
    print_fre = 50#200#1
    gpu = [0]
    test_number = 0
    test_only = True
    # -----------------------------------  tricks ------------------------------------ #
    # already test in pytorch ×7
    multi_lr = False
    warm_up = False
    focus_loss = False
    pseudo_label = False
    all_data = False
    small_batch_size = False
    no_bias_decay = False
    
    # already test in keras ×3
    #label_smoothing 
    #model_embedding
    #6 * TTA
    
    # TODO:(not test ×6)
    #cosine lr decay
    #mix_up
    #center loss
    #autoaugmention
    #autoML
    #zero Gamma

    
# Load the trained model
def load_model(model_path):
    net = SelfDenseNet(argsModel)
    model = torch.nn.DataParallel(net,argsModel.gpu).cuda()
    # model = SelfDenseNet(args)  # Assuming args is globally accessible or pass it as an argument
    checkpoint = torch.load(model_path)
    weight_state = checkpoint['model_state']
    model.load_state_dict(weight_state)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

def preprocess_images_csv(image_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Convert numpy array to PIL Image
    image_list = [Image.fromarray((img * 255).astype(np.uint8).squeeze()) for img in image_data]

    # Apply transformations to each image
    tensor_list = [transform(image) for image in image_list]

    # Stack tensors into a single tensor
    tensor_stack = torch.stack(tensor_list)

    return tensor_stack

def classify_image(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()
