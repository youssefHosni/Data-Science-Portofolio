# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:54:05 2021

@author: youssef Hosni
"""
import pandas as pd
import numpy  as np
import sys

import tensorflow.compat.v1 as tf

sys.path.insert(0,'D:/work & study/Nawah/Datasets/codes/loading and storing')
from loading_images import load_images_from_folder
sys.path.insert(0,'D:/work & study/Nawah/Datasets/codes/preprocessing')
from exploration import  bar_plot,class_counts_proportions
from preprocessing import splitting_normalization
from preprocessing import splitting_classes
sys.path.insert(0,'D:/work & study/Nawah/Datasets/codes/deep learning models')
from main import select_CNN_model
sys.path.insert(0,'D:/work & study/Nawah/Datasets/codes/deep learning models')
from training import training_model 


print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.config.list_physical_devices('GPU') else '\t\u2022 GPU device not found. Running on CPU')

#%%  Loading the png data and split and normalize it 
images_dir = "D:\\work & study\\Nawah\\Datasets\\ISIC_2019_Training_Input\\ISIC_2019_Training_Input"
width = 224
height = 224
input_data = load_images_from_folder(images_dir,width,height)
#%% preprocessiing 
labels = pd.read_csv("D:/work & study/Nawah/Datasets/ISIC_2019_Training_GroundTruth.csv")
labels=labels.iloc[:,1:]
labels.head()
#%%  splitting the data and normalizing it 
train_data,train_labels, val_data,val_labels,test_data,test_labels = splitting_normalization(
                                                                        input_data, 
                                                                        labels
                                                                        )

#%% dividing the data into datasets one with two classes and one with 8 classes 
[train_data_small_classes, 
train_labels_small_classes,
train_data_labels_two_classes] = splitting_classes(train_data,train_labels)

[val_data_small_classes, 
val_labels_small_classes,
val_data_labels_two_classes] = splitting_classes(val_data,val_labels)

[test_data_small_classes, 
test_labels_small_classes,
test_data_labels_two_classes] = splitting_classes(test_data,test_labels)


#%% underesampling the data 
sys.path.insert(0,'D:/work & study/Nawah/Datasets/codes/preprocessing')
from preprocessing import resampling
resampling_stragey = {1:4500}
resampled_training_data, resampled_labels = resampling(
    train_data,
    train_labels,
    'under_sampling',
    resampling_stragey
    )
resampled_training_data = resampled_training_data.reshape(
    resampled_training_data.shape[0],
    224,
    224,
    3
    )
resampled_labels=pd.DataFrame(resampled_labels)
resampled_labels.columns = train_labels.columns[0:-1]
resampled_labels['UNK'] = 0

#%% Oversampling the data
resampling_stragey = {2:2000,4:1700,5:1700,6:2000}
resampled_training_data_small, resampled_labels_small= resampling(
                                        train_data_small_classes,train_labels_small_classes,
                                       'over_sampling',resampling_stragey
                                        )
resampled_training_data_small = resampled_training_data_small.reshape(
                                resampled_training_data_small.shape[0],
                                224,224,3
                                )
resampled_labels_small = pd.DataFrame(resampled_labels_small)
resampled_labels_small.columns=train_labels_small_classes.columns[0:-1]
resampled_labels_small['UNK']=0
#%% Building the simple CNN model and trainning it
models_name_list = [
    'simple_CNN',
    'MobileNet',
    'VGG-16',
    'Inception-v3',
    'InceptionResNetV2'
    ]
model_name=models_name_list[3]
is_trainable=False
epoch_num=100
batch_num=32
evaluation_metrics_list=[
    'accuracy',
    'f1',
    'f1_micro'
    ]
evaluation_metric = evaluation_metrics_list[2]
model = select_CNN_model(model_name,8,is_trainable, np.shape(resampled_training_data_small))
training_model(model,resampled_training_data_small,resampled_labels_small,val_data_small_classes,val_labels_small_classes,test_data_small_classes,
               test_labels_small_classes,epoch_num,batch_num,8,evaluation_metric)
