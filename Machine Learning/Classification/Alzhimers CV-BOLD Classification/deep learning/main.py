import CNN
import load_data
from numpy import load
import numpy as np
import data_preprocessing
import preprocessing_methods
import generate_result_
import os
from scipy.signal import resample_poly 

def main():
   
    
   
    train_data_path='/data/fmri/Folder/AD_classification/Data/input_data/preprocessed_data/CV_OULU_Con_AD_preprocessed.npz'
    train_data_classifer = load(train_data_path)['masked_voxels']
    train_data_path='/data/fmri/Folder/AD_classification/Data/input_data/Augmented_data/CV_OULU_Con_AD_aug.npz'
    train_data_CNN = load(train_data_path)['masked_voxels']
    test_data_path='/data/fmri/Folder/AD_classification/Data/input_data/CV_ADNI_Con_AD.npz'
    test_data_CNN = load(test_data_path)['masked_voxels']
    test_data_path='/data/fmri/Folder/AD_classification/Data/input_data/preprocessed_data/CV_ADNI_Con_AD_preprocessed.npz'
    test_data_classifer = load(test_data_path)['masked_voxels']
    
    transposing_order=[3,0,2,1]
    train_data_CNN=data_preprocessing.transposnig(train_data_CNN,transposing_order)
    test_data_CNN=data_preprocessing.transposnig(test_data_CNN,transposing_order)
    
    train_labels_path='/data/fmri/Folder/AD_classification/Data/input_data/labels/train_labels_aug_data.npz'
    train_labels_CNN=load(train_labels_path)['labels']
    shuffling_indicies = np.random.permutation(len(train_labels_CNN))
    temp = train_data_CNN[shuffling_indicies, :, :, :]
    train_data_CNN=temp
    train_labels_CNN = train_labels_CNN[shuffling_indicies]
    
    
    
    
    
    train_labels_path='/data/fmri/Folder/AD_classification/Data/input_data/labels/train_labels.npz'
    train_labels_classifer=load(train_labels_path)['labels']
    shuffling_indicies = np.random.permutation(len(train_labels_classifer))
    temp = train_data_classifer[shuffling_indicies, :, :, :]
    train_data_classifer=temp
    train_labels_classifer = train_labels_classifer[shuffling_indicies]

    #test_data_path = load_data.find_path(test_data_file_name)
    #test_data_path='/data/fmri/Folder/AD_classification/Data/input_data/CV_ADNI_Con_AD.npz'
    #test_data = load(test_data_path)['masked_voxels']
    #test_labels_path=load_data.find_path(test_labels_file_name)
    test_labels_path='/data/fmri/Folder/AD_classification/Data/input_data/labels/test_labels.npz'
    test_labels=load(test_labels_path)['labels']
    shuffling_indicies = np.random.permutation(len(test_labels))
    test_data_CNN = test_data_CNN[shuffling_indicies, :, :, :]
    test_data_classifer = test_data_classifer[shuffling_indicies, :, :, :]

    test_labels = test_labels[shuffling_indicies]
    
    train_data_CNN,test_data_CNN,train_labels_CNN,test_labels=preprocessing_methods.preprocessing(train_data_CNN,test_data_CNN,train_labels_CNN,test_labels,4,0,None,None)
    
    factors=[(224,45),(224,45),(3,54)]
    train_data_CNN=resample_poly(train_data_CNN, factors[0][0], factors[0][1], axis=1)
    train_data_CNN=resample_poly(train_data_CNN, factors[1][0], factors[1][1], axis=2)
    #train_data_CNN=resample_poly(train_data_CNN, factors[2][0], factors[2][1], axis=3)

    test_data_CNN=resample_poly(test_data_CNN, factors[0][0], factors[0][1], axis=1)
    test_data_CNN=resample_poly(test_data_CNN, factors[1][0], factors[1][1], axis=2)
    #test_data_CNN=resample_poly(test_data_CNN, factors[2][0], factors[2][1], axis=3)


    train_CNN=0
    feature_extraction=1
    
    if train_CNN==1 and feature_extraction==1:
        line1='CNN model is trained and saved and then used as feature extractor'
        line2='CNN model used for feature extraction is :'
    elif train_CNN==1 and feature_extraction==0:
        line1 ='CNN model is trained and used to test the test data'
        line2='CNN model used is :'
    elif train_CNN==0 and feature_extraction==1:
        line1 ='using a saved model to extract fetaures'
        line2='The model used used is a saved model'
    else:
        print('Value Error: train_CNN and feature_extraction cannnot have these values')
        
            
    results_directory='Results'
    num_classes=2
    epoch=1000
    batch_size_factor=1
    optimizer='adam'
    CNN_models=['VGG16','VGG19']
    #intermedidate_layer=[7,7,7,16]
    hyperparameters={'dropouts':[0.25,0.5,0.5],'activation_function':['relu','relu','relu','sigmoid'],'epoch':10,'opt':'adam','penalty':'l1','C':100,'neighbors':50}
    data={'train_data':train_data_CNN,'test_data':test_data_CNN,'train_labels':train_labels_CNN,'test_labels':test_labels}
    preprocessing_method='method 4'
    i=0
    for CNN_model in CNN_models:
        result_path = generate_result_.create_results_dir(results_directory)
        print(CNN_model)
        feature_extractor_parameters={'data':data,'hyperparameters':hyperparameters,'model_type':'pretrained','CNN_model':CNN_model,'intermediate_layer':7,'classifer_name':'all'}
        CNN.CNN_main(train_data_CNN,test_data_CNN,result_path,train_labels_CNN,test_labels,num_classes,epoch,batch_size_factor,optimizer,CNN_model,train_CNN,feature_extraction,feature_extractor_parameters)  
        f = open(os.path.join(result_path, 'README'), "w+")
    
        line3=CNN_model 
        line4='The preprocessing methods used is '+'  '+preprocessing_method
        line5='The number of epochs used to train the CNN_model is '+str(epoch)
        line6='the oprimizer used is '+optimizer
        f.write("{}" "\n" "{}" "\n"  "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" .format(line1,line2,line3,line4,line5,line6))    
        i=i+1
        
if __name__=='__main__':
     main()
     
     
     