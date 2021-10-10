import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#from sklearn import decomposition
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import decomposition
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from scipy import ndimage
import nilearn
import nibabel as nib
import numpy as np
import os 
import load_data

def load_obj(obj):
 # Load subjects
 in_img = nib.load(obj)
 in_shape = in_img.shape
 print('Shape: ', in_shape)
 in_array = in_img.get_fdata()
 return in_array

    

def flipping(img,axis):
    flipped_img = np.flip(img,axis=axis)
    return flipped_img



def flipping_HV(img):
    flipped_img = np.fliplr(img)
    return flipped_img

def rotate(img,angle):
    
    img=ndimage.interpolation.rotate(img,angle)    
    return img

def shifting(img,shift_amount):
    
    img=ndimage.interpolation.shift(img,shift_amount)
    
    return img

def zooming(img,zooming_amount):
    img=ndimage.interpolation.zoom(img,zooming_amount)
    return img


def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col,depth,number_of_samples = X_imgs.shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gaussian_noise_imgs=np.empty(X_imgs.shape)
    for i in range(number_of_samples):
        gaussian_img=np.zeros((row,col,depth))
        gaussian = np.random.random((row, col, 1)).astype(np.float64)
        gaussian = np.tile(gaussian,(1,1,depth))       
        gaussian_img = cv2.addWeighted(X_imgs[:,:,:,i], 0.75, 0.25 * gaussian, 0.25, 0 ,dtype=cv2.CV_64F)
        gaussian_noise_imgs[:,:,:,i]=gaussian_img
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs
  

def transposnig(input_data,order):
    return input_data.transpose(order)



def mask_print(input,mask,name):
 #remained_feature_indices=np.where(mask==1)
    masking_img = nib.load('/data/fmri/Folder/AD_classification/Data/input_data/4mm_brain_mask_bin.nii.gz')

    masking_shape = masking_img.shape
    print(masking_shape)
    masking = np.empty(masking_shape, dtype=float)
    masking[:,:,:] = masking_img.get_data().astype(float)
    for i in range (np.shape(input)[3]):
     input[:,:,:,i]=mask*input[:,:,:,i]
     #input[:,:,:,i]=input[:,:,:,i]
    hdr = masking_img.header
    aff = masking_img.affine
    out_img = nib.Nifti1Image(input, aff, hdr)
     # Save to disk
    out_file_name = '/data/fmri/Folder/AD_classification/Data/input_data/Augmented_data/mask_'+name+'.nii.gz'
    nib.save(out_img, out_file_name)
    
def slicing(len1,len2 ):
    diff=abs(len2-len1)/2 
   
    if (round(diff)>diff):
           return round(diff),len2-round(diff)+1
 
    else:    
        return int(diff),int(len2-diff)


'''
Oulu_data_ad_path = '/data/fmri/Folder/AD_classification/Data/Raw_data/Oulu_Data/CV_OULU_AD.nii.gz'
Oulu_data_con_path='/data/fmri/Folder/AD_classification/Data/Raw_data/Oulu_Data/CV_OULU_CON.nii.gz'
adni_data_ad_path='/data/fmri/Folder/AD_classification/Data/Raw_data/ADNI_Data/CV_ADNI_AD.nii.gz'
adni_data_con_path='/data/fmri/Folder/AD_classification/Data/Raw_data/ADNI_Data/CV_ADNI_CON.nii.gz'
masking_data='/data/fmri/Folder/AD_classification/Data/input_data/4mm_brain_mask_bin.nii.gz'



Oulu_data_ad=load_obj(Oulu_data_ad_path)
Oulu_data_con=load_obj(Oulu_data_con_path)
adni_data_ad=load_obj(adni_data_ad_path)
adni_data_con=load_obj(adni_data_con_path)
mask=load_obj(masking_data)

order_data=(0,2,1,3)
order_mask=(0,2,1)

Oulu_data_con_transposed=transposnig(Oulu_data_con,order_data)
Oulu_data_ad_transposed=transposnig(Oulu_data_ad,order_data)
mask_transposed=transposnig(mask,order_mask)


#Rotation
angles=[30,-30,60,-60,45,-45]
for i in angles:
    Oulu_data_ad_rotated=rotate(Oulu_data_ad_transposed,i)
    mask_rotated=rotate(mask_transposed,i)
    start,end=slicing(Oulu_data_ad.shape[0],Oulu_data_ad_rotated.shape[0])
    
    Oulu_data_ad_rotated=Oulu_data_ad_rotated[0:Oulu_data_ad.shape[0],0:Oulu_data_ad.shape[0],:,:]
    mask_rotated=mask_rotated[0:Oulu_data_ad.shape[0],0:Oulu_data_ad.shape[0],:]

    Oulu_data_ad_rotated_transposed=transposnig(Oulu_data_ad_rotated,order_data)
    mask_rotated_transposed=transposnig(mask_rotated,order_mask)
    print(Oulu_data_ad_rotated_transposed.shape)
    mask_print(Oulu_data_ad_rotated_transposed,mask_rotated_transposed,'rotated_' +str(i)+ '_Oulu_data_ad')



# adding gussian noise 

Oulu_data_ad_noised=add_gaussian_noise(Oulu_data_ad_transposed)
Oulu_data_ad_noised_transposed=transposnig(Oulu_data_ad_noised,order_data)
mask_print(Oulu_data_ad_noised_transposed,mask,'Oulu_data_ad_gussian_noised')


#shifting 
shift_amount_data=[0,20,0,0]
shift_amount_mask=[0,20,0]
Oulu_data_con_shifted=shifting(Oulu_data_con_transposed,shift_amount_data)
Oulu_data_con_shifted_transposed=transposnig(Oulu_data_con_shifted,order_data)
mask_shifted=shifting(mask_transposed,shift_amount_mask)
mask_shifted_transposed=transposnig(mask_shifted,order_mask)
mask_print(Oulu_data_con_shifted_transposed,mask_shifted_transposed,'down_Oulu_data_con')


# flipping
Oulu_data_ad_flipped=flipping(Oulu_data_ad_transposed,0)
Oulu_data_ad_flipped_transposed=transposnig(Oulu_data_ad_flipped,order_data)
mask_tranposed=transposnig(mask,order_mask)
mask_flipped=flipping(mask_tranposed,1)
mask_flipped_transposed=transposnig(mask_flipped,order_mask)
mask_print(Oulu_data_ad_flipped_transposed,mask_flipped_transposed,'vertical_flipped_Oulu_data_ad')
'''


