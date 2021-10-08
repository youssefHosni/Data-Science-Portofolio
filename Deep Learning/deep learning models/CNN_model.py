# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:00:48 2021

@author: youss
"""
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import  Dense, BatchNormalization, Dropout, Activation

# https://keras.io/api/applications/
def simple_CNN(train_data_shape,n_classes):
    # building a linear stack of layers with the sequential model

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=train_data_shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('sigmoid'))
    
    model.summary()
    return model

def MobileNet(num_classes, is_trainable ):   
    
    pretrained_model=tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=False,
        weights="imagenet")

    for layer in pretrained_model.layers[0:18]:
        layer.trainable = is_trainable

    model = Sequential()
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # softmax classifier
    model.add(Dense(num_classes,activation='softmax'))
    pretrainedInput = pretrained_model.input
    pretrainedOutput = pretrained_model.output
    output = model(pretrainedOutput)
    model = tf.keras.models.Model(pretrainedInput, output)
    model.summary()  
    return model 

def VGG_16(num_classes,is_trainable):
    from tensorflow.keras.applications.vgg16 import VGG16

    pretrained_model = VGG16(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet')
    
    for layer in pretrained_model.layers:
        layer.trainable = is_trainable

    model = Sequential()
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # softmax classifier
    model.add(Dense(num_classes,activation='softmax'))
    pretrainedInput = pretrained_model.input
    pretrainedOutput = pretrained_model.output
    output = model(pretrainedOutput)
    model = tf.keras.models.Model(pretrainedInput, output)
    model.summary()  
    return model 

def Inception_v3(num_classes,is_trainable):
    pretrained_model= tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling='max')
    for layer in pretrained_model.layers[0:150]:
        layer.trainable = is_trainable
    model = Sequential()
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # softmax classifier
    model.add(Dense(num_classes,activation='softmax'))
    pretrainedInput = pretrained_model.input
    pretrainedOutput = pretrained_model.output
    output = model(pretrainedOutput)
    model = tf.keras.models.Model(pretrainedInput, output)
    model.summary()  
    return model 

def InceptionResNetV2(num_classes,is_trainable):
    pretrained_model=tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224,224,3))
    
    for layer in pretrained_model.layers[0:450]:
        layer.trainable = is_trainable
    model = Sequential()
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    

    # softmax classifier
    model.add(Dense(num_classes,activation='softmax'))
    pretrainedInput = pretrained_model.input
    pretrainedOutput = pretrained_model.output
    output = model(pretrainedOutput)
    model = tf.keras.models.Model(pretrainedInput, output)
    model.summary()  
    return model 
    
        
        
    