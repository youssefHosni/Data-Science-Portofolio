#from hyper_opt import create_mask,model,model_1D
import load_data
import data_preprocessing
import generate_result
from Model import create_mask
from pathlib import Path



def main():
     train_Con_file_name = 'whole_brain_Oulu_Con.npz'
     train_AD_file_name = 'whole_brain_Oulu_AD.npz'
     test_Con_file_name = 'whole_brain_ADNI_Con.npz'
     test_AD_file_name = 'whole_brain_ADNI_AD.npz'
     root_dir='/data'
     mask_name='4mm_brain_mask_bin.nii.gz'
     results_directory='Results'
     results_path=load_data.find_path(results_directory)
     number_of_cv=5
     feature_selection_type='recursion'
     data_preprocessing_method='kstest and standarization and Normalization and Density ratio estimation'
     Hyperparameter=(4000,10)
     train_data,train_labels=load_data.train_data(train_Con_file_name,train_AD_file_name)
     test_data, test_labels = load_data.test_data(test_Con_file_name, test_AD_file_name)
     #sample_weight = data_preprocessing.density_ratio_estimation(train_data,test_data)
     original_mask=load_data.mask(mask_name,root_dir)

     #created_mask,model_,model_name,weights=create_mask(train_data,labels_train,number_of_cv,feature_selection_type,
                                                         #Hyperparameter,mask_threshold=2,model_type='gaussian_process')

     #test_data = data_preprocessing.coefficient_of_variance(test_data)
     
     #model_, model_name=model_1D(train_data,labels_train,created_mask,data_validation=None,labels_validation=None,model_type='gaussian_process')
     train_data,test_data=data_preprocessing.KSTest(train_data,test_data,step=Hyperparameter[1])


     train_data = data_preprocessing.standarization(train_data)
     test_data = data_preprocessing.standarization(test_data)
     train_data = data_preprocessing.standarization(train_data)
     test_data = data_preprocessing.standarization(test_data)

     train_data = data_preprocessing. MinMax_scaler(train_data)
     test_data= data_preprocessing. MinMax_scaler(test_data)

     sample_weights=data_preprocessing.density_ratio_estimation(train_data,test_data)

     created_mask,model,model_name,weights =create_mask(train_data,train_labels,number_of_cv,feature_selection_type,Hyperparameter[0],1,model_type='Random_forest', sample_weights=sample_weights)
     
     generate_result.print_result(test_data, test_labels, original_mask, created_mask, model, model_name, weights,
                         results_path,feature_selection_type,Hyperparameter,data_preprocessing_method)






if __name__=='__main__':
     main()