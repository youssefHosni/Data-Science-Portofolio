# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 04:58:28 2021

@author: youss
"""
import sys 

sys.path.insert(0,'D:/work & study/Nawah/Datasets/codes/deep learning models')
from CNN_model import Inception_v3
from CNN_model import VGG_16
from CNN_model import simple_CNN
from CNN_model import MobileNet
from CNN_model import InceptionResNetV2

def select_CNN_model(model_name,num_classes,trainable,input_shape):
    
    if model_name == 'simple_CNN':
        model= simple_CNN(input_shape,num_classes)
    
    elif model_name=='MobileNet':
        model=MobileNet(num_classes,trainable)
    
    elif model_name=='VGG-16':
        model=VGG_16(num_classes,trainable)
    
    elif model_name=='Inception-v3':
        model=Inception_v3(num_classes,trainable)
    
    elif model_name=='InceptionResNetV2':
        model=InceptionResNetV2(num_classes,trainable)
    
    else:
        print("Error value : There is no model with the following name",model_name)
        return 
    
    return model
        
def getLayerIndexByName(model, layername):
    
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
    return None    
        