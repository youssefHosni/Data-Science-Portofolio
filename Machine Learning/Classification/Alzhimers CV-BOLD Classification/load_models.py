import pickle
import load_data
import data_preprocessing
import numpy as np
import nibabel as nib
import generate_result

#define paths
train_Con_file_name = 'CV_OULU_CON.npz'
train_AD_file_name = 'CV_OULU_AD.npz'
test_Con_file_name = 'CV_ADNI_CON.npz'
test_AD_file_name = 'CV_ADNI_AD.npz'
mask_name = '4mm_brain_mask_bin.nii.gz'
created_mask_high_certainity_file_name='./Output_results_directory/2019-08-10/1/high_certainity_model_mask.nii.gz'
created_mask_outlier_file_name='./Output_results_directory/2019-08-10/1/high_certainity_model_mask.nii.gz'
high_certainity_model_name='./Output_results_directory/2019-08-10/1/high_certainity_model.sav'
low_certainty_model_name='./Output_results_directory/2019-08-10/1/low_certainty_model.sav'
outliers_model_name='./Output_results_directory/2019-08-10/1/outliers_model.sav'
#define variables
number_of_neighbours = 1

#load data
train_data,train_labels=load_data.train_data_3d(train_Con_file_name,train_AD_file_name)
test_data, test_labels = load_data.test_data_3d(test_Con_file_name, test_AD_file_name)

#load masks
mask_4mm = load_data.mask(mask_name)
created_mask_high_certainity = nib.load(created_mask_high_certainity_file_name)
created_mask_outlier = nib.load(created_mask_outlier_file_name)


#data preprocessing
train_data = np.moveaxis(train_data.copy(), 3, 0)
test_data = np.moveaxis(test_data.copy(), 3, 0)
original_mask=mask_4mm.get_fdata()
train_data = train_data * original_mask
test_data = test_data * original_mask
created_mask_high_certainity=created_mask_high_certainity.get_fdata()
created_mask_outlier=created_mask_outlier.get_fdata()
orignal_mask_flatten = data_preprocessing.flatten(original_mask[np.newaxis, :, :, :].copy())
orignal_mask_flatten = np.reshape(orignal_mask_flatten, (-1))
created_mask_high_certainity_flatten = data_preprocessing.flatten(created_mask_high_certainity[np.newaxis, :, :, :].copy())
created_mask_high_certainity_flatten = np.reshape(created_mask_high_certainity_flatten, (-1))
created_mask_outlier_flatten = data_preprocessing.flatten(created_mask_outlier[np.newaxis, :, :, :].copy())
created_mask_outlier_flatten = np.reshape(created_mask_outlier_flatten, (-1))
train_data_flattened = data_preprocessing.flatten(train_data.copy())
test_data_flattened = data_preprocessing.flatten(test_data.copy())
train_data_flattened = data_preprocessing.MinMax_scaler(train_data_flattened.copy())
test_data_flattened = data_preprocessing.MinMax_scaler(test_data_flattened.copy())


train_data_inlier, train_labels_inlier, outlier_indices_train = data_preprocessing.outliers(train_data_flattened,
                                                                                                train_labels,
                                                                                                number_of_neighbours)
test_data_inlier, test_labels_inlier, outlier_indices_test = data_preprocessing.novelty(train_data_inlier,
                                                                                            train_labels_inlier,
                                                                                            test_data_flattened,
                                                                                            test_labels,
                                                                                            number_of_neighbours)

test_data_inlier_brain=test_data_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
test_data_outlier_brain=(test_data_flattened[outlier_indices_test])[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
test_data_masked_high_certainity=test_data_inlier_brain* created_mask_high_certainity_flatten[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)]
test_data_inlier_CVspace = data_preprocessing.coefficient_of_variance(test_data_masked_high_certainity)[:,np.newaxis]
test_data_outlier_cv = data_preprocessing.coefficient_of_variance(
        test_data_outlier_brain *created_mask_outlier_flatten[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:, np.newaxis]
#load models
high_certainity_model = pickle.load(open(high_certainity_model_name, 'rb'))
low_certainty_model = pickle.load(open(low_certainty_model_name, 'rb'))
outliers_model = pickle.load(open(outliers_model_name, 'rb'))
#output results
test_accuracy_high_certainity,F1_score_high_certainity,auc_high_certainity,low_confidence_indices=generate_result.out_result_highprob(test_data_inlier_CVspace,
                                                                                                                test_labels_inlier,orignal_mask_flatten,created_mask_high_certainity_flatten,high_certainity_model)
test_accuracy_low_certainty,F1_score_low_certainty,auc_low_certainty=generate_result.out_result(test_data_inlier_CVspace[low_confidence_indices],
                                                                                                 test_labels_inlier[low_confidence_indices],orignal_mask_flatten,created_mask_high_certainity_flatten,low_certainty_model)
test_accuracy_outlier,F1_score_outlier,auc_outlier= generate_result.out_result(test_data_outlier_cv ,
                                                                   test_labels[outlier_indices_test], orignal_mask_flatten,
                                                                   created_mask_outlier_flatten, outliers_model)


#print results
print('total_test_accuracy>',(test_accuracy_high_certainity+test_accuracy_low_certainty+test_accuracy_outlier)/3)
print('total_F1_score>',(F1_score_high_certainity+F1_score_low_certainty+F1_score_outlier)/3)
print('total_AUC_score>',(auc_high_certainity+auc_low_certainty+auc_outlier)/3)