from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from scipy import ndimage
import nilearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from scipy.stats import variation
import numpy as np
from densratio import densratio
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import resample
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE,ADASYN
from sklearn import preprocessing
from scipy.stats import ks_2samp
import tensorflow as tf
import math
import load_data
import data_augmentation
from scipy.signal import resample_poly


def Normalization(data):
    return Normalizer().fit_transform(data)


def standarization(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data

def quantile_transform(data,random_state):
    quantile_transformer = preprocessing.QuantileTransformer(random_state=random_state)
    data = quantile_transformer.fit_transform(data)
    return data

def gussian_filter(data,sigma):
    for i in range(len(data)):
        data[i] = ndimage.gaussian_filter(data[i], sigma)

    return data

def signal_clean(data):
    data = nilearn.signal.clean(data)

    return data
def robust_scaler(data):
    scaler = RobustScaler()
    data = scaler.fit_transform(data)

    return data

def MinMax_scaler(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    return data

def dublicate(data,number):
    for i in range(number):
        data=np.vstack((data,data))

    return data

def concat(data1,data2):

    data1=np.vstack((data1,data2))

    return data1

def shuffling(data,labels):
    idx = np.random.permutation(len(labels))
    data, labels = data[idx], labels[idx]

    return data,labels

def PowerTransform(data):
     power_transform = PowerTransformer()
     data=power_transform.fit_transform(data)
     
     return data

def coefficient_of_variance(data):
    #data=MinMax_scaler(data)
    data = variation(data, axis=1)

    return data

def density_ratio_estimation(train_data,test_data):
    result = densratio(train_data,test_data)
    sample_weight=result.compute_density_ratio(train_data)  

    return sample_weight

def outliers(train_data,train_labels,number_of_neighbours):
    neigh = LocalOutlierFactor(n_neighbors=number_of_neighbours)
    indices=neigh.fit_predict(train_data)
    train_data_inlier=train_data[np.where(indices==1)]
    train_labels_inlier=train_labels[np.where(indices==1)]
    outlier_indices=np.where(indices==-1)

    return train_data_inlier,train_labels_inlier,outlier_indices

def novelty(train_data,train_labels,test_data,test_labels,number_of_neighbours):
    neigh = LocalOutlierFactor(n_neighbors=number_of_neighbours,novelty=True)
    indices=neigh.fit(train_data)
    indices=indices.predict(test_data)
    test_data_inlier=test_data[np.where(indices==1)]
    test_labels_inlier=test_labels[np.where(indices==1)]
    outlier_indices=np.where(indices==-1)
    
    return test_data_inlier,test_labels_inlier,outlier_indices

def upsampling(data,labels):
    X = np.hstack((data, labels))
    not_fewsamples = X[np.where(X[:, -1] == 0)]

    fewsamples = X[np.where(X[:, -1] == 1)]
    fewsamples_upsampled = resample(fewsamples,
                                    replace=False,  # sample with replacement
                                    n_samples=len(not_fewsamples)-len(fewsamples),  # match number in majority class
                                    random_state=42)  # reproducible results
    fewsamples_upsampled=np.vstack((fewsamples_upsampled, fewsamples))
    fewsamples_upsampled = np.vstack((fewsamples_upsampled, not_fewsamples))
    fewsamples_upsampled = shuffle(fewsamples_upsampled, random_state=42)
    labels = fewsamples_upsampled[:, -1]
    data = fewsamples_upsampled[:, 0:np.shape(fewsamples_upsampled)[1] - 1]

    return data,labels

def synthetic(data,labels,num):
    smote = ADASYN(ratio='all',n_neighbors=num)
    data, labels = smote.fit_sample(data, labels)
    
    return data,labels

def KSTest(train_data,test_data,step):

    index=[]
    for i in range(0,len(train_data)-step,step):
        for j in range(train_data.shape[1]):

            r=ks_2samp(train_data[i:i+step,j],test_data[:,j])
            if r[1]>0.05:
                index=np.append(index,j)
    print(train_data.shape)
    if index==[]:
        return train_data,test_data
    index=index[:,np.newaxis]
    index=index.astype(int)
    index=removeDuplicates(index)
    train_data[:,index]=0
    test_data[:,index]=0
    

    return train_data,test_data


def removeDuplicates(listofElements):

    # Create an empty list to store unique elements
    uniqueList = []

    # Iterate over the original list and for each element
    # add it to uniqueList, if its not already there.
    for elem in listofElements:
        if elem not in uniqueList:
            uniqueList.append(elem)

    # Return the list of unique elements
    return uniqueList

def transposnig(input_data, order):
    return input_data.transpose(order)


def size_editing(data, final_height):
    data_length = data.shape[1]
    if (data_length > final_height):
        diff = abs(data_length - final_height) / 2
        if (round(diff) > diff):
            start = round(diff)
            end = data_length - round(diff) + 1
            return data[:, start:end, start:end, :]

        else:
            start = int(diff)
            end = int(data_length - diff)
            return data[:, start:end, start:end, :]
    else:
        diff = abs(data_length - final_height) / 2
        if (round(diff) > diff):
            resized_data=np.pad(data,((0,0),(round(diff),round(diff)-1),(round(diff),round(diff)-1),(0,0)),'constant',constant_values=(0, 0))
        else:
            resized_data=np.pad(data,((0,0),(round(diff),round(diff)),(round(diff),round(diff)),(0,0)),'constant',constant_values=(0, 0))
        
        return resized_data
  
def depth_reshapeing(data):

    depth=int(data.shape[3])

    dim0=int(data.shape[0])

    dim1=int(data.shape[1])

    dim2=int(data.shape[2])

    step=math.floor(depth/3)

    reshaped_data=np.empty((dim0,dim1,dim2,3))
   
    for i in range(3):
        
        if i==2:
            reshaped_data[:,:,:,i]=np.mean(data[:,:,:,step*i:depth],axis=3) 
        else:            
            reshaped_data[:,:,:,i]=np.mean(data[:,:,:,step*i:step*(i+1)],axis=3)
    return reshaped_data    

def converting_nii_to_npz(file_path):
    #file_path=load_data.find_path(file_name)
    nii_file=data_augmentation.load_obj(file_path)
    np.savez(file_path[0:len(file_path)-7]+'.npz',masked_voxels=nii_file)   


def labels_convert_one_hot(labels):
    length=len(labels)
    zeros=np.zeros((length,1))
    labels=np.hstack((zeros,labels))
    indecies=np.where(labels[:,1]==0)
    labels[indecies[0],0]=1
    
                
    
    return labels   

def data_resampling(data,factors):
    
    for k in range(len(factors)):
        data = resample_poly(data, factors[k][0], factors[k][1], axis=k+1)
        
        
        
        
    
    