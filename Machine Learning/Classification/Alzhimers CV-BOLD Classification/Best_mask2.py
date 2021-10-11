#This file is for obtaining confidence interval of the used procedure and to get the best mask that specifies the maximum AUC

import numpy as np
from hyper_opt import create_mask,model_reduced,model_1D,model_1D_calibrate
import load_data
import data_preprocessing
import generate_result
from sklearn.model_selection import train_test_split

def main():
    # define input file names, directories, and parameters
    train_Con_file_name = 'CV_con.npz'
    train_AD_file_name = 'CV_pat.npz'
    #test_Con_file_name = 'CV_ADNI_CON.npz'
    #test_AD_file_name = 'CV_ADNI_AD.npz'
    mask_name = '4mm_brain_mask_bin_epl.nii.gz'
    feature_selection_type = 'L2_penality'
    results_directory = 'newResults'
    results_path = load_data.find_path(results_directory)
    from sklearn.model_selection import train_test_split

    #defining variables
    number_of_cv = 5
    Hyperparameter_model__1 = 1000
    Hyperparameter_model__3 = 1000
    number_of_neighbours = 1
    np.random.seed(1)
    accuracy_total_list = list()
    F1_score_total_list = list()
    auc_total_list = list()
    Best_AUC = 0
    model_name='decison tree classifer'

    # loading input data and mask
    train_data,train_labels=load_data.train_data_3d(train_Con_file_name,train_AD_file_name)
    #test_data, test_labels = load_data.test_data_3d(test_Con_file_name, test_AD_file_name)
    mask_4mm = load_data.mask(mask_name)
    original_mask=mask_4mm.get_fdata()


    # data preprocessing
    train_data = np.moveaxis(train_data.copy(), 3, 0)
    #test_data = np.moveaxis(test_data.copy(), 3, 0)
    train_data = train_data * original_mask
    #test_data = test_data * original_mask
    shape = np.shape(train_data)
    train_data_flattened = data_preprocessing.flatten(train_data.copy())
    #test_data_flattened = data_preprocessing.flatten(test_data.copy())
    orignal_mask_flatten = data_preprocessing.flatten(original_mask[np.newaxis, :, :, :].copy())
    orignal_mask_flatten = np.reshape(orignal_mask_flatten, (-1))
    train_data_flattened = data_preprocessing.MinMax_scaler(train_data_flattened.copy())
    #test_data_flattened = data_preprocessing.MinMax_scaler(test_data_flattened.copy())
    
    #confidence_interval using bootstraping
    for _ in range(100):
        number_of_neighbours = 1
        train_data_flattened, test_data_flattened, train_labels, test_labels = train_test_split(train_data_flattened, train_labels, test_size=0.2, random_state=42)
        train_data_inlier, train_labels_inlier, outlier_indices_train = data_preprocessing.outliers(train_data_flattened,
                                                                                              train_labels,
                                                                                              number_of_neighbours)
        if len(train_data_inlier)<=1: 
                continue
        test_data_inlier, test_labels_inlier, outlier_indices_test = data_preprocessing.novelty(train_data_inlier,
                                                                                            train_labels_inlier,
                                                                                            test_data_flattened,
                                                                                            test_labels,
                                                                                            number_of_neighbours)

        train_labels_inlier=train_labels_inlier[:, np.newaxis]
        #train_data_outliers, trian_labels_outliers = data_preprocessing.resampling(train_data_flattened[outlier_indices_train].copy(),
        #                                                                          train_labels[outlier_indices_train].copy())
        train_data_outliers, trian_labels_outliers =train_data_flattened[outlier_indices_train],(train_labels[outlier_indices_train])[:, np.newaxis]

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
        if len(train_data_outliers)<=1: 
                continue
        train_data_outlier_inlier, train_labels_outlier_inlier, outlier_outlier_indices_train = data_preprocessing.novelty(train_data_outliers, trian_labels_outliers,
                                                                                                                           train_data_outlier_more_flattened,
                                                                                                                           train_labels_outlier_more,
                                                                                                                           number_of_neighbours)
	
        train_data_inlier, train_labels_inlier = data_preprocessing.upsampling(train_data_inlier,train_labels_inlier)
        if len(train_data_inlier)==0: continue

        train_data_inlier, train_labels_inlier = data_preprocessing.shuffling(train_data_inlier,train_labels_inlier)
        print('np.shape(train_data_outlier_inlier)',np.shape(train_data_outlier_inlier))
        print('np.shape(train_data_outlier_inlier)[0]',np.shape(train_data_outlier_inlier)[0])
        if np.shape(train_data_inlier)[0]>0:

                train_data_outlier_inlier, train_labels_outlier_inlier = data_preprocessing.upsampling(
                                                                                                train_data_outlier_inlier.copy(),
                                                                                                train_labels_outlier_inlier.copy())
                train_data_outlier_inlier, train_labels_outlier_inlier = data_preprocessing.shuffling(train_data_outlier_inlier.copy(),
                                                                                                train_labels_outlier_inlier.copy())
        if len(train_data_outlier_inlier)==0: continue

        


        #Brain extraction of data
        train_data_inlier_brain=train_data_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        test_data_inlier_brain=test_data_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        train_data_outlier_inlier_brain=train_data_outlier_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        test_data_outlier_brain=(test_data_flattened[outlier_indices_test])[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
        concated_data = data_preprocessing.concat(train_data_inlier, train_data_outlier_inlier)
        print('train_labels_inlier[:, np.newaxis]',np.shape(train_labels_inlier[:, np.newaxis]))
        print('train_labels_outlier_inlier[:, np.newaxis]',np.shape(train_labels_outlier_inlier[:, np.newaxis]))
        if (len(np.shape(train_labels_outlier_inlier[:, np.newaxis]))<=2):
                concated_labels = data_preprocessing.concat(train_labels_inlier[:, np.newaxis],train_labels_outlier_inlier[:, np.newaxis])
        else:
                concated_labels = data_preprocessing.concat(train_labels_inlier[:, np.newaxis],train_labels_outlier_inlier)
        #Model stage 1 with high certainity
        model1_created_mask, model1_, model1_name, model1_weights = create_mask(train_data_inlier_brain, train_labels_inlier,
                                                                                number_of_cv, feature_selection_type,
                                                                                Hyperparameter_model__1, mask_threshold=4,
                                                                                model_type='gaussian_process')
        #train_data_inlier_CVspace = data_preprocessing.coefficient_of_variance(train_data_inlier_brain * model1_created_mask)[:,np.newaxis]
        #test_data_inlier_CVspace = data_preprocessing.coefficient_of_variance(test_data_inlier_brain * model1_created_mask)[:,np.newaxis]
        #model1_, model1_name = model_1D(train_data_inlier_CVspace, train_labels_inlier, model1_created_mask,
        #                                data_validation=None, labels_validation=None,
        #                                model_type='gaussian_process')
        train_data_inlier_CVspace = (train_data_inlier_brain * model1_created_mask)
        test_data_inlier_CVspace = (test_data_inlier_brain * model1_created_mask)

        model1_, model1_name = model_reduced(train_data_inlier_CVspace, train_labels_inlier, model1_created_mask,
                                              data_validation=None, labels_validation=None,
                                              model_type='gaussian_process')

        model1_test_accuracy,model1_F1_score,model1_auc,low_confidence_indices=generate_result.out_result_highprob(test_data_inlier_CVspace,
                                                                                                                   test_labels_inlier,
                                                                                                                   original_mask,model1_created_mask,
                                                                                                                   model1_)

        #Model stage 2 with low certainity
        if (low_confidence_indices!=0):
            model2_, model2_name = model_reduced(train_data_inlier_CVspace, train_labels_inlier, model1_created_mask,
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
        model3_created_mask, model3_, model3_name, model3_weights = create_mask(concated_data,
                                                                                concated_labels, number_of_cv,
                                                                                feature_selection_type, Hyperparameter_model__3,
                                                                                mask_threshold=4,
                                                                                model_type='gaussian_process')
        #concated_data_cv = data_preprocessing.coefficient_of_variance(
        #    concated_data[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)].copy() * model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:, np.newaxis]
        #test_data_outlier_cv = data_preprocessing.coefficient_of_variance(test_data_outlier_brain *model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:, np.newaxis]
        #model3_, model3_name = model_1D(concated_data_cv, concated_labels, model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)],
        #                                data_validation=None, labels_validation=None, model_type='gaussian_process')
        
        #model3_test_accuracy,model3_F1_score,model3_auc = generate_result.out_result(np.nan_to_num(test_data_outlier_cv) ,
        #                                                                             test_labels[outlier_indices_test], np.nan_to_num(original_mask),
        #                                                                             np.nan_to_num(model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)]), model3_)
        concated_data_cv = (concated_data[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)].copy() * 
                         model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])
        test_data_outlier_cv = (test_data_outlier_brain *model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])

        model3_, model3_name = model_reduced(concated_data_cv, concated_labels, model3_created_mask,
                                             data_validation=None, labels_validation=None, model_type=model_name)
        model3_test_accuracy,model3_F1_score,model3_auc = generate_result.out_result(np.nan_to_num(test_data_outlier_cv) ,
                                                                                 np.nan_to_num(test_labels[outlier_indices_test]), np.nan_to_num(original_mask),
                                                                                 np.nan_to_num(model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)]), model3_)

        #stacking procedure of bootstapping and printing models
        div = 3
        print('np.shape(test_labels_inlier[low_confidence_indices])',np.shape(test_labels_inlier[low_confidence_indices]))
        print('length(test_labels_inlier[low_confidence_indices])',len(test_labels_inlier[low_confidence_indices]))
        print('np.shape(test_labels_inlier[low_confidence_indices])[0]',np.shape(test_labels_inlier[low_confidence_indices])[0])
        if np.logical_and((model1_test_accuracy==0),(len(test_labels_inlier[low_confidence_indices])==len(test_labels_inlier))): div=div-1
        if np.logical_and((model3_test_accuracy==0),(len(test_labels[outlier_indices_test]))==0): div=div-1
        testnum=len(test_labels)
        highcernum=(len(test_labels_inlier)-len(test_labels_inlier[low_confidence_indices]))/testnum
        lowcernum=(len(test_labels_inlier[low_confidence_indices]))/testnum
        outnum=(len(test_labels[outlier_indices_test]))/testnum
        avg_accuracy = (highcernum*model1_test_accuracy + lowcernum*model2_test_accuracy + outnum*model3_test_accuracy) 
        avg_F1_score = (highcernum*model1_F1_score + lowcernum*model2_F1_score + outnum*model3_F1_score) 
        avg_auc = (highcernum*model1_auc + lowcernum*model2_auc + outnum*model3_auc) 
        accuracy_total_list.append(avg_accuracy)
        F1_score_total_list.append(avg_F1_score)
        auc_total_list.append(avg_auc)
        if (avg_auc>Best_AUC):
            Best_AUC=avg_auc
            data_preprocessing_method = "Seperating outlier of training set and test set, then synthethise more data from training-outliers, then appling probability predictions. High probability " \
                                        "samples model is used with predictions with high probability, then apply low probability model. Finally add noise to outliers and concatinate with inlier data" \
                                        "to be used for outlier model"
            generate_result.print_result_3models(mask_4mm, results_path, model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)], model3_, model3_name,
                                                 model3_weights[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)],model3_test_accuracy,model3_auc, model3_F1_score, Hyperparameter_model__3,
                                                 model2_, model2_name, model2_test_accuracy, model2_auc,model2_F1_score,
                                                 model1_, model1_created_mask, model1_name, model1_weights,model1_test_accuracy, model1_auc, model1_F1_score,Hyperparameter_model__1,
                                                 feature_selection_type, data_preprocessing_method,highcernum,lowcernum,outnum)

    # total_performance_confidence_level
    generate_result.confidence_interval_model_99(accuracy_total_list, F1_score_total_list, auc_total_list, 'total_performance')



if __name__=='__main__':
     main()