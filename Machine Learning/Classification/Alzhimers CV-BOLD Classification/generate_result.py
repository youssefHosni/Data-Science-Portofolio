import numpy as np
import os
import nibabel as nib
from datetime import date
from sklearn import metrics
from sklearn.metrics import f1_score
import load_data
'''
test_Con_file_name='whole_brain_ADNI_Con.npz'
test_AD_file_name='whole_brain_ADNI_AD.npz'
root_dir='/data'
mask_name='4mm_brain_mask_bin.nii.gz'
results_directory=load_data.find_path('results','/data')
mask,model,model_name,weights=Model.create_mask()
'''
def out_result(test_data,test_labels,original_mask,created_mask,model):
    if (np.shape(test_data)[1]==1):
         test_data=np.reshape(test_data,(-1,1))
         predicted_labels = model.predict(test_data)
         test_accuracy = model.score(test_data,test_labels[:,np.newaxis])
    else:
         masked_test_data = test_data*created_mask
         predicted_labels = model.predict(masked_test_data[:,np.squeeze(np.where(created_mask>0),axis=0)])
         test_accuracy = model.score(masked_test_data[:,np.squeeze(np.where(created_mask>0),axis=0)],test_labels[:,np.newaxis])

    F1_score = f1_score(test_labels,predicted_labels, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
    auc=metrics.auc(fpr, tpr)
    return test_accuracy,F1_score,auc
def print_result_2models(test_data,test_labels,original_mask,model,model_name,results_directory,model1_accuracy,model1_auc,model1_f1,model1_name,model1_mask,model1_weights,feature_selection_type,Hyperparameter,data_preprocessing_method):
    if (np.shape(test_data)[1]==1):
         test_data=np.reshape(test_data,(-1,1))
         predicted_labels = model.predict(test_data)
         test_accuracy = model.score(test_data,test_labels[:,np.newaxis])
    else:
         masked_test_data = test_data*model1_mask
         predicted_labels = model.predict(masked_test_data[:,np.squeeze(np.where(model1_mask>0),axis=0)])
         test_accuracy = model.score(masked_test_data[:,np.squeeze(np.where(model1_mask>0),axis=0)],test_labels[:,np.newaxis])

    F1_score = f1_score(test_labels,predicted_labels, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
    auc=metrics.auc(fpr, tpr)


    today = str(date.today()) # To save the results in a directory with the date as a name

    if os.path.exists(os.path.join(results_directory,today))==0:
        os.mkdir(os.path.join(results_directory,today))

    if len(os.listdir(os.path.join(results_directory,today))) ==0:
        file_number = 1
    else:

        #latest_file = sorted(os.path.join(results_directory,today),key=x,reverse=True)
        print(os.path.join(results_directory, today))
        dir_list=os.listdir(os.path.join(results_directory,today))
        latest_file=sorted(list(map(int,dir_list)),reverse=True)
        print(latest_file)

        file_number = ((latest_file[0]))+1

    os.mkdir(os.path.join(results_directory,today,str(file_number)))

    line1 = 'Test accuracy on inliers:' + '  ' + str(model1_accuracy)
    line2 = 'F1 score on inliers :' + '  ' + str(model1_f1)
    line3 = 'AUC on inliers :' + '  ' + str(model1_auc)
    line4 = 'Test accuracy on outliers :' + '  ' + str(test_accuracy)
    line5 = 'F1 score on outliers :' + '  ' + str(F1_score)
    line6 = 'AUC on outliers :' + '  ' + str(auc)
    line7 = 'Total test accuracy :' + '  ' + str((model1_accuracy+test_accuracy)/2)
    line8 = 'Total F1_score :' + '  ' + str((model1_f1 + F1_score) / 2)
    line9 = 'Total AUC :' + '  ' + str((model1_auc + auc) / 2)
    f = open(os.path.join(results_directory,today,str(file_number),'Results.txt'),"w+")
    f.write("{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" .format(line1, line2, line3,line4,line5,line6,line7,line8,line9))


    line1='The model used to obtain first model result is '+ '  '+ model1_name
    line2='The model used to obtain second result is '+ '  '+ model_name
    line3='The feature selection methods is ' + '  '+ feature_selection_type
    line4= 'the hyperparameter used is ' + '  '+ str(Hyperparameter)
    line5= 'The preprocessing method used is '+ ' '+ data_preprocessing_method
    f=open(os.path.join(results_directory,today,str(file_number),'README.txt'),"w+")
    f.write("{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n"  .format(line1,line2,line3,line4,line5))

    mask_print(original_mask,model1_mask,os.path.join(results_directory,today,str(file_number)),'model_')
    weights_print(original_mask ,model1_weights, os.path.join(results_directory,today,str(file_number)),'model_')

    return
def print_result(test_data,test_labels,original_mask,created_mask,model,model_name,weights,results_directory,feature_selection_type,Hyperparameter,data_preprocessing_method):


    masked_test_data = test_data*created_mask
    print(masked_test_data.shape)
    print(masked_test_data[:,np.squeeze(np.where(created_mask>0),axis=0)].shape)

    predicted_labels = model.predict(masked_test_data[:,np.squeeze(np.where(created_mask>0),axis=0)])

    print(predicted_labels)

    test_accuracy = model.score(masked_test_data[:,np.squeeze(np.where(created_mask>0),axis=0)],test_labels[:,np.newaxis])
    F1_score = f1_score(test_labels,predicted_labels, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
    auc=metrics.auc(fpr, tpr)


    today = str(date.today()) # To save the results in a directory with the date as a name

    if os.path.exists(os.path.join(results_directory,today))==0:
        os.mkdir(os.path.join(results_directory,today))

    if len(os.listdir(os.path.join(results_directory,today))) ==0:
        file_number = 1
    else:

        #latest_file = sorted(os.path.join(results_directory,today),key=x,reverse=True)
        print(os.path.join(results_directory, today))
        dir_list=os.listdir(os.path.join(results_directory,today))
        latest_file=sorted(list(map(int,dir_list)),reverse=True)
        print(latest_file)

        file_number = ((latest_file[0]))+1

    os.mkdir(os.path.join(results_directory,today,str(file_number)))

    line1 = 'Test accuracy is:' + '  ' + str(test_accuracy)
    line2 = 'F1 score is:' + '  ' + str(F1_score)
    line3 = 'AUC is :' + '  ' + str(auc)
    f = open(os.path.join(results_directory,today,str(file_number),'Results.txt'),"w+")
    f.write("{}" "\n" "{}" "\n" "{}" "\n"  .format(line1, line2, line3))


    line1='The model used to obtain this result is '+ '  '+ model_name
    line2='The feature selection methods is ' + '  '+ feature_selection_type
    line3= 'the hyperparameter used is ' + '  '+ str(Hyperparameter)
    line4= 'The preprocessing method used is '+ ' '+ data_preprocessing_method
    f=open(os.path.join(results_directory,today,str(file_number),'README.txt'),"w+")
    f.write("{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" .format(line1,line2,line3,line4))
    mask_print(original_mask,created_mask,os.path.join(results_directory,today,str(file_number)),'output_')
    weights_print(original_mask ,weights, os.path.join(results_directory,today,str(file_number)),'output_')
    return


def mask_print(original_mask,created_mask,output_dir,name):

    masking_shape = original_mask.shape
    masking = np.empty(masking_shape, dtype=float)
    masking[:, :, :] = original_mask.get_data().astype(float)
    masking[np.where(masking > 0)] = masking[np.where(masking > 0)] * 0 + created_mask
    hdr = original_mask.header
    aff = original_mask.affine
    out_img = nib.Nifti1Image(masking, aff, hdr)
    nib.save(out_img, os.path.join(output_dir,name+'mask.nii.gz'))
    return

def weights_print(original_mask, weights, output_dir,name):

    masking_shape = original_mask.shape
    masking = np.empty(masking_shape, dtype=float)
    masking[:, :, :] = original_mask.get_data().astype(float)
    masking[np.where(masking > 0)] = masking[np.where(masking > 0)] * 0 + weights
    hdr = original_mask.header
    aff = original_mask.affine
    out_img = nib.Nifti1Image(masking, aff, hdr)
    nib.save(out_img, os.path.join(output_dir, name+'weights.nii.gz'))
    return


def cnn_save_result(test_accuracy,model,model_name,result_path):
  
    line1 = 'Test accuracy is:' + '  ' + str(test_accuracy)
    f = open(os.path.join(result_path, 'Results.txt'), "w+")
    f.write("{}" "\n" .format(line1))

    model.save(os.path.join(result_path, 'trained_model.h5'))

    f = open(os.path.join(result_path, 'README'), "w+")
    line1 ='CNN model was used '
    line2 = 'The model used to obtain this result is ' + '  ' + model_name
    f.write("{}" "\n" "{}" "\n" .format(line1,line2))



def create_results_dir(results_directory):
    today = str(date.today())  # To save the results in a directory with the date as a name

    Results_dir_path=load_data.find_path(results_directory)
    if os.path.exists(os.path.join(Results_dir_path, today)) == 0:
        os.mkdir(os.path.join(Results_dir_path, today))

    if len(os.listdir(os.path.join(Results_dir_path, today))) == 0:
        file_number = 1
    else:

        # latest_file = sorted(os.path.join(results_directory,today),key=x,reverse=True)
        print(os.path.join(Results_dir_path, today))
        dir_list = os.listdir(os.path.join(Results_dir_path, today))
        latest_file = sorted(list(map(int, dir_list)), reverse=True)
        print(latest_file)
        file_number = ((latest_file[0])) + 1
    os.mkdir(os.path.join(Results_dir_path, today, str(file_number)))
    result_path=os.path.join(Results_dir_path, today, str(file_number))
    print(result_path)
    return result_path
