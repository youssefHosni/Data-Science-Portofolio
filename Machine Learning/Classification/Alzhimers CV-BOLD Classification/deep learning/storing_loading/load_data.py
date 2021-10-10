import numpy as np
from numpy import load
import os
import nibabel as nib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def find_path(file_name):
    data_path=None
    current_dir_path=os.getcwd()
    p=Path(current_dir_path)
    root_dir=p.parts[0]+p.parts[1]
    for r,d,f in os.walk(root_dir):
        for files in f:
             if files == file_name:
                data_path=os.path.join(r,files)

             else:
                 for dir in d :
                    if dir == file_name:
                        data_path = os.path.join(r, dir)
    if data_path is not None:
        return data_path
    else:
        os.makedirs('./'+file_name)
        return './'+file_name



def train_data_3d(train_Con_file_name, train_AD_file_name):
    train_data_Con_path = find_path(train_Con_file_name)
    train_data_AD_path = find_path(train_AD_file_name)
    train_data_Con = load(train_data_Con_path)['masked_voxels']
    train_data_AD = load(train_data_AD_path)['masked_voxels']
    train_data=np.concatenate((train_data_Con,train_data_AD),axis=3)
    train_labels = np.hstack((np.zeros(train_data_Con.shape[3]), np.ones(train_data_AD.shape[3])))


    return train_data, train_labels

def test_data_3d(test_Con_file_name,test_AD_file_name):
    test_data_Con_path=find_path(test_Con_file_name)
    test_data_AD_path = find_path(test_AD_file_name)
    test_data_Con = load(test_data_Con_path)['masked_voxels']
    test_data_AD = load(test_data_AD_path)['masked_voxels']
    test_data = np.concatenate((test_data_Con, test_data_AD), axis=3)
    test_labels = np.hstack((np.zeros(test_data_Con.shape[3]), np.ones(test_data_AD.shape[3])))


    return test_data,test_labels


def mask(mask_name):
    mask_path = find_path(mask_name)
    original_mask = nib.load(mask_path)
    return original_mask

