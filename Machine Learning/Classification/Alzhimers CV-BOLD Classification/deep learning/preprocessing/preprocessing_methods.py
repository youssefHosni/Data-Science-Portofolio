import data_preprocessing
import numpy as np
import load_data

def preprocessing(train_data,test_data,train_labels,test_labels,method,save,file_name,output_dir):        
    
   
    
    dim0_train=train_data.shape[0]
    dim1_train=train_data.shape[1]
    dim2_train=train_data.shape[2]
    dim3_train=train_data.shape[3]
    
    dim0_test=test_data.shape[0]
    dim1_test=test_data.shape[1]
    dim2_test=test_data.shape[2]
    dim3_test=test_data.shape[3]
      
    if method==0:
        return
    elif method==1:
        train_data=train_data.reshape(dim0_train,dim1_train*dim2_train*dim3_train)
        test_data=test_data.reshape(dim0_test,dim1_test*dim2_test*dim3_test)
        train_data=data_preprocessing.MinMax_scaler(train_data)
        test_data=data_preprocessing.MinMax_scaler(test_data)
        train_data,test_data=data_preprocessing.standarization(train_data,test_data)
        train_data,test_data=data_preprocessing.KSTest(train_data,test_data,800)
        train_data=train_data.reshape(dim0_train,dim1_train,dim2_train,dim3_train)
        test_data=test_data.reshape(dim0_test,dim1_test,dim2_test,dim3_test)
            
    elif method==2:
        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[3]):
                train_data[i, :, :, j] = data_preprocessing.standarization(train_data[i, :, :, j])
                train_data[i, :, :, j] = data_preprocessing.MinMax_scaler(train_data[i, :, :, j])
            
                if i < test_data.shape[0]:
                    test_data[i, :, :, j] = data_preprocessing.standarization(test_data[i, :, :, j])
        
                    test_data[i, :, :, j] = data_preprocessing.MinMax_scaler(test_data[i, :, :, j]) 
        

    
    
    
    
    elif method==3: 
        train_data=train_data.reshape(dim0_train,dim1_train*dim2_train*dim3_train)
        test_data=test_data.reshape(dim0_test,dim1_test*dim2_test*dim3_test)
        train_data,test_data=data_preprocessing.KSTest(train_data,test_data,500)                                     
        train_data=train_data.reshape(dim0_train,dim1_train,dim2_train,dim3_train)
        test_data=test_data.reshape(dim0_test,dim1_test,dim2_test,dim3_test)
        

  
    elif method==4:    
        train_data=train_data.reshape(dim0_train,dim1_train*dim2_train,dim3_train)
        test_data=test_data.reshape(dim0_test,dim1_test*dim2_test,dim3_test)
        for i in range (dim3_train):    
            train_data[:,:,i]=data_preprocessing.MinMax_scaler(train_data[:,:,i])
            train_data[:,:,i]=data_preprocessing.standarization(train_data[:,:,i])
            
            test_data[:,:,i]=data_preprocessing.MinMax_scaler(test_data[:,:,i])
            test_data[:,:,i]=data_preprocessing.standarization(test_data[:,:,i])
            
            train_data[:,:,i],test_data[:,:,i]=data_preprocessing.KSTest(train_data[:,:,i],test_data[:,:,i],800)       
        train_data=train_data.reshape(dim0_train,dim1_train,dim2_train,dim3_train)
        test_data=test_data.reshape(dim0_test,dim1_test,dim2_test,dim3_test)
       
    elif method==5:
        train_data=train_data.reshape(dim0_train,dim1_train*dim2_train*dim3_train)
        train_data,train_labels,index=data_preprocessing.outliers(train_data,train_labels,1)
        train_data=train_data.reshape(dim0_train-np.size(index),dim1_train,dim2_train,dim3_train)
    if save==0:
        
        return train_data,test_data,train_labels,test_labels
    else:
        
        transposing_order = [1,3,2,0]
        train_data = data_preprocessing.transposnig(train_data, transposing_order)
        test_data = data_preprocessing.transposnig(test_data, transposing_order)
        output_path=load_data.find_path(output_dir)
        np.savez(output_path+file_name+'train_data.npz',masked_voxels=train_data)
        np.savez(output_path+file_name+'test_data.npz',masked_voxels=test_data)
    
