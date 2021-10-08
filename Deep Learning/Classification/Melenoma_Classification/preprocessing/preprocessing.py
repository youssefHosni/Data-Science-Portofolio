# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:20:13 2021
@author: youss
"""

import numpy as np
import time
import cv2

from multiprocessing.dummy import Pool
from multiprocessing.sharedctypes import Value
from ctypes import c_int
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks,NearMiss
from imblearn.under_sampling import OneSidedSelection

def splitting_normalization(input_data, labels):
    train_data,val_data,train_labels,val_labels = train_test_split(input_data,labels, test_size=0.3, random_state=42)
    val_data,test_data,val_labels,test_labels = train_test_split(val_data,val_labels, test_size=0.33, random_state=42)
    
    # building the input vector from the 28x28 pixels
    train_data=np.array(train_data)
    val_data=np.array(val_data)
    test_data=np.array(test_data)
    
    train_data = train_data.astype('float32')
    val_data = val_data.astype('float32')
    test_data = test_data.astype('float32')
    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)
    
    # normalizing the data to help with the training
    train_data /= 255
    val_data /= 255
    test_data /= 255
    return train_data,train_labels, val_data,val_labels,test_data,test_labels



def splitting_classes(input_data,input_labels):
    
    """
    Parameters
    ----------
    input_data : Array 
        The input data with all classes .
    input_labels : DataFrame 
        The input labels of data with all classes.

    Returns
    -------
    ouput_data_small_classes : Array 
        The input data with all classes .
    output_labels_small_classes : DataFrame 
        The input labels of data with all classes.
    output_labels_two_classes : DataFrame 
        The input labels of data with all classes.

    """
    input_labels.reset_index(drop=True,inplace=True)
    output_labels_small_classes=input_labels[input_labels['NV']==0]
    output_labels_small_classes.drop(columns='NV',inplace=True)
    ouput_data_small_classes=input_data[output_labels_small_classes.index.values]
    
    
    output_labels_two_classes=input_labels.copy()
    output_labels_two_classes['other_classes']=0
    output_labels_two_classes.iloc[output_labels_small_classes.index.values,9]=1

    labels_to_drop_index=[0,2,3,4,5,6,7,8]
    output_labels_two_classes.drop(columns=output_labels_two_classes.columns[labels_to_drop_index],inplace=True)

    return ouput_data_small_classes, output_labels_small_classes,output_labels_two_classes

def resizing_data(input_data,width, height):
    resized_data=[]
    def read_imagecv2(img, counter):
        img = cv2.resize(img, (width, height))
        resized_data.append(img)
        with counter.get_lock(): #processing pools give no way to check up on progress, so we make our own
            counter.value += 1

    # start 4 worker processes
    with Pool(processes=2) as pool: #this should be the same as your processor cores (or less)
        counter = Value(c_int, 0) #using sharedctypes with mp.dummy isn't needed anymore, but we already wrote the code once...
        chunksize = 4 #making this larger might improve speed (less important the longer a single function call takes)
        resized_test_data = pool.starmap_async(read_imagecv2, ((img, counter) for img in input_data) ,  chunksize) #how many jobs to submit to each worker at once  
        while not resized_test_data.ready(): #print out progress to indicate program is still working.
            #with counter.get_lock(): #you could lock here but you're not modifying the value, so nothing bad will happen if a write occurs simultaneously
            #just don't `time.sleep()` while you're holding the lock
            print("\rcompleted {} images   ".format(counter.value), end='')
            time.sleep(.5)
        print('\nCompleted all images')
    return resized_data  



def resampling(train_data,train_labels,resampling_type,resampling_stragey):
    train_data_new=np.reshape(train_data,(train_data.shape[0],train_data.shape[1]*train_data.shape[2]*train_data.shape[3]))
    if resampling_type == 'SMOTE':
        train_data_resampled,train_labels_resampled = SMOTE(random_state=42).fit_resample(train_data_new, train_labels.values)
   
    elif resampling_type=='over_sampling':
        over_sampler=RandomOverSampler(sampling_strategy=resampling_stragey)
        train_data_resampled, train_labels_resampled = over_sampler.fit_resample(train_data_new,train_labels.values) 
    
    elif resampling_type== 'under_sampling':
        under_sampler=RandomUnderSampler(sampling_strategy=resampling_stragey)
        train_data_resampled, train_labels_resampled = under_sampler.fit_resample(train_data_new,train_labels.values)
    
    elif resampling_type == 'tomelinks':
        t1= TomekLinks( sampling_strategy=resampling_stragey)
        train_data_resampled, train_labels_resampled = t1.fit_resample(train_data_new,train_labels.values )
    
    elif resampling_type=='near_miss_neighbors':
        undersample = NearMiss(version=1, n_neighbors=3)
        train_data_resampled, train_labels_resampled = undersample.fit_resample(train_data_new,train_labels.values )
    
    elif resampling_type=='one_sided_selection':
        undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
        train_data_resampled, train_labels_resampled = undersample.fit_resample(train_data_new,train_labels.values )
    
    return train_data_resampled, train_labels_resampled 




