import numpy as np
from numpy import load
import os


oulu_con_data=load('/data/fmri/Folder/AD_classification/Data/input_data/whole_brain_Oulu_Con.npz')['masked_voxels']
oulu_ad_data=load('/data/fmri/Folder/AD_classification/Data/input_data/whole_brain_Oulu_AD.npz')['masked_voxels']
adni_con_data=load('/data/fmri/Folder/AD_classification/Data/input_data/whole_brain_ADNI_Con.npz')['masked_voxels']
adni_ad_data=load('/data/fmri/Folder/AD_classification/Data/input_data/whole_brain_ADNI_AD.npz')['masked_voxels']







idx = np.random.permutation(np.shape(oulu_con_data)[0])
oulu_con_data= (oulu_con_data)[idx,:]
oulu_ad_data=(oulu_ad_data)[idx,:]
adni_con_data=(adni_con_data)[idx,:]
adni_ad_data=(adni_ad_data)[idx,:]

print(idx)
print(np.shape(idx))
os.mkdir('./data')
np.savez('./data/oulu_con_data', masked_voxels=oulu_con_data)
np.savez('./data/oulu_ad_data',masked_voxels=oulu_ad_data)
np.savez('./data/adni_con_data',masked_voxels=adni_con_data)
np.savez('./data/adni_ad_data',masked_voxels=adni_ad_data)
np.savez('./data/key',idx)

npzfile = np.load('data/key.npz')
npzfile=np.asarray(npzfile['arr_0'])
print(npzfile)
print(np.shape(npzfile))