'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import os
import sys
from numpy import load
import keras
import load_data
import generate_result_
import data_preprocessing
import CNN_feature_extractor
import LeNet
import simple_model
import DenseNet121
import InceptionResNetV2
import ResNet50
import VGG_pretrained
import VGG
import ZFNet
import AlexNet
import optimizers


def CNN_main(train_data,test_data,result_path,train_labels,test_labels,num_classes,epoch,batch_size_factor,optimizer,CNN_model,train_CNN,feature_extraction,feature_extractor_parameters):
    '''
    Function Description: This function is the main function for the CNN modules and it controls them all,
    it decides which method to be used , CNN as a classifer or as a feature extractor and for each one of 
    them which CNN model and architetcture to be used and also the paramteres for each one of the case is 
    defined by the variables as described later.
    
    Function Parameters:
        -------------------------------------------------------
        Train_data_file_name:The name of the train data file , this will be differ depeneding on the CNN model 
        used,For the following models 'ResNet50','inception_model,'DenseNet121','VGG_pretrained' a resized data 
        to diminssion 224*224*3 will be used , this one was resized from the original data and was saved after, 
        as this conversion takes alot of time so as to decrease the computation time.
        Test_file_name:The name of the test data file and there are manily two files as mentioned in the previous
        variable.
        results_directory:the directory name where you would like the output files to be generated
        train_data_file_name:This is used to load the training data labels , also you should choose the
        right file as there are different training data files due to differnet augmentation methods used 
        and they differ in their number so the lables will differ also
        test_data_file_name: the name of the test labels file , this will also be the same as long as  the 
        test data is not changable.
        num_classes:the number of the classes the our data will be classifed to.
        epoch:the number of iter through all the training dataset that the CNN model will go through
        batch_size_factor:this will take the batch as a factor of the training data size and since the
        trainig data in our case is small , then it will be alawys =1
        optimizer:this will be the optimizer used for training the CNN model, the variable should have 
        the name of one of the optimizer in the code , else error will occur.
        CNN_model:this define which CNN model to be trained on the training dataset or it will be None if a saved
        model will be used as feature extractor 
        train_CNN: This selects one of the two modes ,=0 means  to train a CNN_model or =1 to use a pretrained model
        or a saved model to extrat features from the training data directly.
       feature_extraction:This is variable wil be used only if train_CNN =0 and it will select between two options;
       =0 to trian the CNN model only without using it for feature extraction and if it is =1 the model after 
       being trained it will be saved and used for feature extraction.
       feature_extractor_parameters:
           This is a dict of four variables and they are used as an input parameters for the feature extraction 
           functionan they are as the following :
                 'CNN_model': this will be the CNN model used for feature extraction, it may takes different type , it
                 may be string , in the case of pretrained model so the string will be equal to the name of the selected model
                 or it may take 'all' in this case all the pretrained models will be used , in case of saved model this can take
                 two forms, the first to be string if the feature_extraction=1 , this mean that a saved model will be used directly
                 without being trained and if feature_extraction=1 it will be th model that have just been trained and saved.
                 
                 'model_type':to choose between the pretrained models on imagenet dataset or the saved trained models
                 on the training set.
                 'classifer_name': this is the classifer name you would like to use , it can be 'all', or one of them.
                 'hyperprametres': this is another dict in which the classifer hyperparmaters is defined, it contain four main
                 parameters , the value of the neigbors to look to for the KNN classifer and the for SVC classifer thera are
                 two hyperparamters 'penalty' and 'C' and for the fully conncted classifer there are four hyperparamters
                 'dropouts','activation_functions','opt','epoch'.If any of this classifers is not used then it's hyperparmater should 
                  be =None.
    '''
    
    opt=optimizers.choosing(optimizer)
    
    if train_CNN==1:
        if CNN_model=='simple_model':
            simple_model.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        elif CNN_model=='LeNet':
            LeNet.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        elif CNN_model=='AlexNet':
            AlexNet.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        elif CNN_model=='ZFNet':
            ZFNet.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        elif CNN_model=='VGG':
            VGG.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        elif CNN_model=='VGG_pretrained':
            VGG_pretrained.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        elif CNN_model=='ResNet50':
            ResNet50.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        elif CNN_model=='inception_model':
            InceptionResNetV2.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)        
        elif CNN_model== 'DenseNet121':
            DenseNet121.model(train_data,train_labels,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters)
        else:
            print('Value Error: CNN_model took unexpected value')
            sys.exit()
    
    
    elif train_CNN==0:
        CNN_feature_extractor.CNN_feature_extraction_classsification(feature_extractor_parameters,result_path)

    else:
        print('Value Error: train_CNN took unexpected value')
        sys.exit()
    
                

