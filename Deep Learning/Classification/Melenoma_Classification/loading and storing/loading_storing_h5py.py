# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:55:49 2021

@author: youss
"""
import numpy as np
import h5py
import os 


def storing_h5py(input_data,hdf5_dir):
    for i in range (len(input_data)):
        image_id=i
        image= input_data[i]
        file = h5py.File(os.path.join(hdf5_dir,str(image_id)+'.h5'), "w")
        dataset = file.create_dataset("image", np.shape(image), h5py.h5t.STD_U8BE, data=image)
        file.close()

def read_h5py(hdf5_dir,num_images):
    images=[]
    for i in range(num_images):
        image_id=i
        file = h5py.File(os.path.join(hdf5_dir,str(image_id)+'.h5'), "r+")
        image = np.array(file["/image"]).astype("uint8")
        images.append(image)
    return images

