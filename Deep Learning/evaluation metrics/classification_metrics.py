# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:57:42 2021

@author: youss
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix_calc(predicted_labels,true_labels,num_classes,title):
    positions = np.arange(0,num_classes)
    classes = np.arange(0,num_classes)
    cm=confusion_matrix(predicted_labels.argmax(axis=1), true_labels.values.argmax(axis=1),labels=classes) 
    disp=ConfusionMatrixDisplay(cm,display_labels=classes)
    classes_name = true_labels.columns.values
    plt.figure(figsize=(10,10))
    disp.plot()
    plt.xticks(positions, classes_name)
    plt.yticks(positions, classes_name)
    plt.title(title)
    return 
    

