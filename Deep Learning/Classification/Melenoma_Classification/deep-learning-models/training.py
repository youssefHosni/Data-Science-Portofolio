# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:31:01 2021

@author: youss
"""
import numpy as np
import sys

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import SGD

sys.path.insert(0,'D:/work & study/Nawah/Datasets/codes/evaluation metrics')
from f1_score import f1,f1_micro
from classification_metrics import confusion_matrix_calc

def training_model(model,train_data,train_labels,val_data,val_labels,test_data,test_labels,num_epoch,batch_size,num_classes,evaulation_metric):
    
    class_weights = class_weight.compute_class_weight('balanced',np.unique(train_labels.values.argmax(axis=1)),train_labels.values.argmax(axis=1))
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    es = EarlyStopping(monitor='val_'+evaulation_metric, mode='max', verbose=1,patience=10,baseline=0.5,min_delta=0.1)
    
    if evaulation_metric=='accuracy':
        model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    elif evaulation_metric=='f1':
        model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=[f1])
    elif evaulation_metric=='f1_micro':
        model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=[f1_micro])
    
        
    history=model.fit(train_data,train_labels, validation_data=(val_data,val_labels),epochs=num_epoch, batch_size=batch_size,class_weight=class_weights,callbacks=[es])
    score=model.evaluate(test_data,test_labels)
    print(f'Test loss: {score[0]} / Test' + ' ' + evaulation_metric + f'score: {score[1]}')
    plotting_train_val_metrics(history,evaulation_metric)    
    predicted_train_labels=model.predict(train_data)
    predicted_val_labels=model.predict(val_data)
    predicted_test_labels=model.predict(test_data)
    confusion_matrix_calc(predicted_train_labels,train_labels,num_classes,'confusion matrix of the training data') 
    confusion_matrix_calc(predicted_val_labels,val_labels,num_classes,'confusion matrix of the validation data') 
    confusion_matrix_calc(predicted_test_labels,test_labels,num_classes,'confusion matrix of the test data') 
    
    return None

def plotting_train_val_metrics(history,evaulation_metric):
    plt.plot(history.history[evaulation_metric])
    plt.plot(history.history['val_'+ evaulation_metric])
    plt.title('training and val' + evaulation_metric)
    plt.ylabel(evaulation_metric +'score')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    return None
      