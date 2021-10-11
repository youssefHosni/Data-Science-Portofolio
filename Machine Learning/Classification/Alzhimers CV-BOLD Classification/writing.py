import numpy as np
from numpy import load
import nibabel as nib

masking_img = nib.load('/data/fmri/Folder/AD_classification/Data/input_data/4mm_brain_mask_bin.nii.gz')
masking_shape = masking_img.shape

masking = np.empty(masking_shape, dtype=float)
masking[:,:,:] = masking_img.get_data().astype(float)
print(masking.shape)
tmp=np.where(masking[2,:,:]>0)
print(((tmp)))



'''
import os 
from datetime import date
import glob

today = date.today()
x='hello'
f=open('test.txt',"w+")
f.write(x+'world')

os.mkdir('writing')
LatestFile = sorted(os.listdir('/data/fmri/Folder/AD_classification/codes/model/writing'),reverse = True)

x=int(LatestFile[0][0])
x=x+1
print(x)

'''