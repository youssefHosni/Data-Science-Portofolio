import pickle
import load_data
import data_preprocessing
import numpy as np
import nibabel as nib
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import variation

train_Con_file_name = 'CV_OULU_CON.npz'
train_AD_file_name = 'CV_OULU_AD.npz'
mask_name = '4mm_brain_mask_bin.nii.gz'
created_mask_high_certainity_file_name='./Output_results_directory/2019-08-10/1/high_certainity_model_mask.nii.gz'
created_mask_outlier_file_name='./Output_results_directory/2019-08-10/1/high_certainity_model_mask.nii.gz'
high_certainity_model_name='./Output_results_directory/2019-08-10/1/high_certainity_model.sav'
low_certainty_model_name='./Output_results_directory/2019-08-10/1/low_certainty_model.sav'
outliers_model_name='./Output_results_directory/2019-08-10/1/outliers_model.sav'
scaler_name='scaler.sav'
number_of_neighbours = 1
model_type='gaussian_process'

#load nii file
sample_name='CV_ADNI_AD.nii.gz'
sample_path=load_data.find_path(sample_name)
sample = nib.load(sample_path)
sample = sample.get_fdata()
sample=sample[:,:,:,7] #comment if using 3d data

#load necessary files
mask_4mm = load_data.mask(mask_name)
original_mask=mask_4mm.get_fdata()
orignal_mask_flatten = data_preprocessing.flatten(original_mask[np.newaxis, :, :, :].copy())
orignal_mask_flatten = np.reshape(orignal_mask_flatten, (-1))
created_mask_high_certainity = nib.load(created_mask_high_certainity_file_name)
created_mask_outlier = nib.load(created_mask_outlier_file_name)
created_mask_high_certainity=created_mask_high_certainity.get_fdata()
created_mask_outlier=created_mask_outlier.get_fdata()
created_mask_high_certainity_flatten = data_preprocessing.flatten(created_mask_high_certainity[np.newaxis, :, :, :].copy())
created_mask_high_certainity_flatten = np.reshape(created_mask_high_certainity_flatten, (-1))
created_mask_outlier_flatten = data_preprocessing.flatten(created_mask_outlier[np.newaxis, :, :, :].copy())
created_mask_outlier_flatten = np.reshape(created_mask_outlier_flatten, (-1))
train_data,train_labels=load_data.train_data_3d(train_Con_file_name,train_AD_file_name)
train_data = np.moveaxis(train_data.copy(), 3, 0)
train_data = train_data * original_mask
train_data_flattened = data_preprocessing.flatten(train_data.copy())
train_data_flattened = data_preprocessing.MinMax_scaler(train_data_flattened.copy())



#preprocessing
sample_pre=np.array(sample)*original_mask
sample_pre=np.reshape(sample_pre,(1,-1))
scaler = pickle.load(open(scaler_name, 'rb'))
sample_pre=scaler.transform(sample_pre)


#load_models
high_certainity_model = pickle.load(open(high_certainity_model_name, 'rb'))
low_certainty_model = pickle.load(open(low_certainty_model_name, 'rb'))
outliers_model = pickle.load(open(outliers_model_name, 'rb'))

#prediction
neigh = LocalOutlierFactor(n_neighbors=number_of_neighbours,novelty=True)
neighbours=neigh.fit(train_data_flattened)
inlier_outlier_state=neighbours.predict(sample_pre.copy())
if (inlier_outlier_state==1):#inlier
    sample_pre = variation(sample_pre[:,np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)]*
                           created_mask_high_certainity_flatten[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)],axis=1)[:,np.newaxis]


    if (model_type!='ensamble classifer'):
        sample_prob=high_certainity_model.predict_proba(sample_pre)
        if ((sample_prob[0,0]>.64)|(sample_prob[0,0]<.35)):
            sample_pred=high_certainity_model.predict(sample_pre)
            print('high certainty prediction')

        else:
            sample_pred = low_certainty_model.predict(sample_pre)
            print('low certainty prediction')

    else:
        sample_pred = low_certainty_model.predict(sample_pre)

else:
    sample_pre=variation(sample_pre[:,np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)]*
                                                          created_mask_outlier_flatten[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)],axis=1)[:,np.newaxis]
    sample_pred = outliers_model.predict(sample_pre)
    print('an outlier prediction')
if sample_pred:
    print('sample-prediction: AD')
else:
    print('sample-prediction: CON')


