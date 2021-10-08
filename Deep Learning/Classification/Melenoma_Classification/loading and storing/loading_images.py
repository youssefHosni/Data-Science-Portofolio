# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 02:40:27 2021

@author: youss
"""
import cv2
import os

    
def load_images_from_folder(folder,width,height):
    images = []
    i=0
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img=cv2.resize(img,(width,height))
        if img is not None:
            images.append(img)
            i=i+1
    return images


