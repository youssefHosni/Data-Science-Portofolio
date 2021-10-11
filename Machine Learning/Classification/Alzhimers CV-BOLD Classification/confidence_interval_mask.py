import numpy as np
from hyper_opt import create_mask,model_1D,model_1D_calibrate
import load_data
import data_preprocessing
import generate_result
import nibabel as nib


def main():
    # define input file names, directories, and parameters
    train_Con_file_name = 'CV_OULU_CON.npz'
    train_AD_file_name = 'CV_OULU_AD.npz'
    test_Con_file_name = 'CV_ADNI_CON.npz'
    test_AD_file_name = 'CV_ADNI_AD.npz'
    mask_name = '4mm_brain_mask_bin.nii.gz'
    created_mask_high_certainity_file_name = './Output_results_directory/2019-08-09/13/high_certainity_model_mask.nii.gz'
    created_mask_outlier_file_name = './Output_results_directory/2019-08-09/13/outliers_model_mask.nii.gz'

    #define variables
    number_of_neighbours = 1
    accuracy_total_list = list()
    F1_score_total_list = list()
    auc_total_list = list()


    # loading input data and mask
    train_data,train_labels=load_data.train_data_3d(train_Con_file_name,train_AD_file_name)
    test_data, test_labels = load_data.test_data_3d(test_Con_file_name, test_AD_file_name)
    mask_4mm = load_data.mask(mask_name)
    original_mask=mask_4mm.get_fdata()
    created_mask_high_certainity = nib.load(created_mask_high_certainity_file_name)
    created_mask_outlier = nib.load(created_mask_outlier_file_name)
    created_mask_high_certainity = created_mask_high_certainity.get_fdata()
    created_mask_outlier = created_mask_outlier.get_fdata()


    # data preprocessing
    train_data = np.moveaxis(train_data.copy(), 3, 0)
    test_data = np.moveaxis(test_data.copy(), 3, 0)
    train_data = train_data * original_mask
    test_data = test_data * original_mask
    shape = np.shape(test_data)
    train_data_flattened = data_preprocessing.flatten(train_data.copy())
    test_data_flattened = data_preprocessing.flatten(test_data.copy())
    orignal_mask_flatten = data_preprocessing.flatten(original_mask[np.newaxis, :, :, :].copy())
    orignal_mask_flatten = np.reshape(orignal_mask_flatten, (-1))
    train_data_flattened = data_preprocessing.MinMax_scaler(train_data_flattened.copy())
    test_data_flattened = data_preprocessing.MinMax_scaler(test_data_flattened.copy())
    created_mask_high_certainity_flatten = data_preprocessing.flatten(
    created_mask_high_certainity[np.newaxis, :, :, :].copy())
    created_mask_high_certainity_flatten = np.reshape(created_mask_high_certainity_flatten, (-1))
    created_mask_outlier_flatten = data_preprocessing.flatten(created_mask_outlier[np.newaxis, :, :, :].copy())
    created_mask_outlier_flatten = np.reshape(created_mask_outlier_flatten, (-1))
    train_data_inlir, train_labels_inlir, outlier_indices_train = data_preprocessing.outliers(train_data_flattened,
                                                                                               train_labels,
                                                                                               number_of_neighbours)
    test_data_inlier, test_labels_inlier, outlier_indices_test = data_preprocessing.novelty(train_data_inlir,
                                                                                             train_labels_inlir,
                                                                                             test_data_flattened,
                                                                                             test_labels,
                                                                                             number_of_neighbours)
    model1_created_mask=created_mask_high_certainity_flatten
    model3_created_mask=created_mask_outlier_flatten
    #confidence_interval using bootstraping
    for _ in range(1000):

        train_data_inlier,train_labels_inlier=data_preprocessing.resampling(train_data_inlir.copy(), train_labels_inlir.copy())
        train_data_outliers, trian_labels_outliers = data_preprocessing.resampling(train_data_flattened[outlier_indices_train].copy(),
                                                                               train_labels[outlier_indices_train].copy())

        train_data_inlier_unflattened = data_preprocessing.deflatten(train_data_inlier, shape)
        train_data_outlier_unflattened = data_preprocessing.deflatten(train_data_outliers, shape)
        train_data_inlier_unflattened = np.moveaxis(train_data_inlier_unflattened.copy(), 0, 3)
        train_data_outlier_unflattened = np.moveaxis(train_data_outlier_unflattened.copy(), 0, 3)
        train_data_inlier_noised = data_preprocessing.apply_noise_manytypes(train_data_inlier_unflattened.copy())
        train_data_inlier_filtered = data_preprocessing.apply_filter_manytypes(train_data_inlier_unflattened.copy())
        train_data_inlier_more = data_preprocessing.concatination(train_data_inlier_noised, train_data_inlier_filtered)
        train_labels_inlier_more = data_preprocessing.dublicate(train_labels_inlier.copy(), 29)
        train_data_outlier_noised = data_preprocessing.apply_noise_manytypes(train_data_outlier_unflattened.copy())
        train_data_outlier_filtered = data_preprocessing.apply_filter_manytypes(train_data_outlier_unflattened.copy())
        train_data_outlier_more = data_preprocessing.concatination(train_data_outlier_noised, train_data_outlier_filtered)
        train_labels_outlier_more = data_preprocessing.dublicate(trian_labels_outliers.copy(), 29)
        train_data_inlier_more = np.moveaxis(train_data_inlier_more.copy(), 3, 0)
        train_data_outlier_more = np.moveaxis(train_data_outlier_more.copy(), 3, 0)
        train_data_inlier_more_flattened = data_preprocessing.flatten(train_data_inlier_more.copy())
        train_data_outlier_more_flattened = data_preprocessing.flatten(train_data_outlier_more.copy())
        # train_data_inlier_inlier, train_labels_inlier_inlier, inlier_outlier_indices_train = data_preprocessing.novelty(
        #     train_data_inlier, train_labels_inlier,
        #     train_data_inlier_more_flattened,
        #     train_labels_inlier_more,
        #     number_of_neighbours)
        train_data_outlier_inlier, train_labels_outlier_inlier, outlier_outlier_indices_train = data_preprocessing.novelty(train_data_outliers, trian_labels_outliers,
                                                                                                                            train_data_outlier_more_flattened,
                                                                                                                            train_labels_outlier_more,
                                                                                                                            number_of_neighbours)
        train_data_inlier, train_labels_inlier = data_preprocessing.upsampling(train_data_inlier,train_labels_inlier)
        if train_data_inlier is None:
            continue

        train_data_inlier, train_labels_inlier = data_preprocessing.shuffling(train_data_inlier,train_labels_inlier)
        train_data_outlier_inlier, train_labels_outlier_inlier = data_preprocessing.upsampling(train_data_outlier_inlier,
                                                                                               train_labels_outlier_inlier)
        if train_data_outlier_inlier is None:
            continue

        train_data_outlier_inlier, train_labels_outlier_inlier = data_preprocessing.shuffling(train_data_outlier_inlier,
                                                                                                train_labels_outlier_inlier)


        #Brain extraction of data
        train_data_inlier_brain=train_data_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        test_data_inlier_brain=test_data_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        train_data_outlier_inlier_brain=train_data_outlier_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        test_data_outlier_brain=(test_data_flattened[outlier_indices_test])[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        concated_data = data_preprocessing.concat(train_data_inlier, train_data_outlier_inlier)
        concated_labels = data_preprocessing.concat(train_labels_inlier[:, np.newaxis],train_labels_outlier_inlier[:,np.newaxis])
        
        #Model stage 1 with high certainity
        train_data_inlier_CVspace = data_preprocessing.coefficient_of_variance(train_data_inlier_brain * model1_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:,np.newaxis]
        test_data_inlier_CVspace = data_preprocessing.coefficient_of_variance(test_data_inlier_brain * model1_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:,np.newaxis]
        model1_, model1_name = model_1D(train_data_inlier_CVspace, train_labels_inlier, model1_created_mask,
                                        data_validation=None, labels_validation=None,
                                        model_type='gaussian_process')
        model1_test_accuracy,model1_F1_score,model1_auc,low_confidence_indices=generate_result.out_result_highprob(test_data_inlier_CVspace,
                                                                                                                    test_labels_inlier,
                                                                                                                    original_mask,model1_created_mask,
                                                                                                                    model1_)
        #Model stage 2 with low certainity
        if (low_confidence_indices!=0):
            model2_, model2_name = model_1D_calibrate(train_data_inlier_CVspace, train_labels_inlier, model1_created_mask,
                                                      data_validation=None, labels_validation=None,
                                                      model_type='gaussian_process')
            model2_test_accuracy, model2_F1_score, model2_auc = generate_result.out_result(test_data_inlier_CVspace[low_confidence_indices],
                                                                                            test_labels_inlier[low_confidence_indices],
                                                                                            original_mask,
                                                                                            model1_created_mask,
                                                                                            model2_)
        else:
            model2_test_accuracy, model2_F1_score, model2_auc=0,0,0
        #Model stage 3 with outliers
        concated_data_cv = data_preprocessing.coefficient_of_variance(concated_data[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)].copy() * 
                                                                       model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:, np.newaxis]
        test_data_outlier_cv = data_preprocessing.coefficient_of_variance(test_data_outlier_brain *
                                                                           model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:, np.newaxis]
        model3_, model3_name = model_1D(concated_data_cv, concated_labels, model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)],
                                        data_validation=None, labels_validation=None, model_type='gaussian_process')
        model3_test_accuracy,model3_F1_score,model3_auc = generate_result.out_result(test_data_outlier_cv ,
                                                                                      test_labels[outlier_indices_test], original_mask,
                                                                                      model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)], model3_)

        #stacking part of bootstrapping
        div = 3
        if (model2_test_accuracy==0):
            div=div-1
        if (model1_test_accuracy==0):
            div=div-1

        avg_accuracy = (model1_test_accuracy  + model2_test_accuracy  + model3_test_accuracy ) / div
        avg_F1_score = (model1_F1_score  + model2_F1_score  + model3_F1_score ) /div
        avg_auc = (model1_auc  + model2_auc  + model3_auc ) / div
        accuracy_total_list.append(avg_accuracy)
        F1_score_total_list.append(avg_F1_score)
        auc_total_list.append(avg_auc)

    # total_performance_confidence_level
    generate_result.confidence_interval_model_95(accuracy_total_list, F1_score_total_list, auc_total_list, 'total_performance')



if __name__=='__main__':
     main()