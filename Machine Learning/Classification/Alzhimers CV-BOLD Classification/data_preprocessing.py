from sklearn.preprocessing import StandardScaler
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
from imblearn.over_sampling import ADASYN
from sklearn import preprocessing
from scipy.stats import ks_2samp
from sklearn.decomposition import FastICA, PCA
import nibabel as nib
from skimage.util import random_noise
from scipy.signal import wiener
from skimage.filters import unsharp_mask
from scipy import signal
import math
import load_data
import data_augmentation


####2D transformation

def standarization(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    return data


def quantile_transform(data, random_state):
    quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=36, random_state=random_state)
    data = quantile_transformer.fit_transform(data)
    return data


def gussian_filter(data, sigma):
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


def dublicate(data, number):
    data_stacked = data.copy()
    for i in range(number):
        data_stacked = np.vstack((data, data_stacked))

    return data_stacked


def concat(data1, data2):
    data1 = np.vstack((data1, data2))

    return data1


def shuffling(data, labels):
    idx = np.random.permutation(len(labels))
    data, labels = data[idx], labels[idx]

    return data, labels


def PowerTransform(data):
    power_transform = PowerTransformer()
    data = power_transform.fit_transform(data)

    return data


def coefficient_of_variance(data):
    data = MinMax_scaler(data)
    data = variation(data, axis=1)

    return data


def density_ratio_estimation(train_data, test_data):
    result = densratio(train_data, test_data)
    sample_weight = result.compute_density_ratio(train_data)

    return sample_weight


def outliers(train_data, train_labels, number_of_neighbours):
    neigh = LocalOutlierFactor(n_neighbors=number_of_neighbours)
    indices = neigh.fit_predict(train_data)
    train_data_inlier = train_data[np.where(indices == 1)]
    train_labels_inlier = train_labels[np.where(indices == 1)]
    outlier_indices = np.where(indices == -1)

    return train_data_inlier, train_labels_inlier, outlier_indices


def novelty(train_data, train_labels, test_data, test_labels, number_of_neighbours):
    neigh = LocalOutlierFactor(n_neighbors=number_of_neighbours, novelty=True)
    indices = neigh.fit(train_data)
    indices = indices.predict(test_data)
    test_data_inlier = test_data[np.where(indices == 1)]
    print('test_labels[np.where(indices == 1)]',np.shape(test_labels[np.where(indices == 1)]))
    test_labels_inlier = test_labels[np.where(indices == 1)]
    outlier_indices = np.where(indices == -1)

    return test_data_inlier, test_labels_inlier, outlier_indices


def upsampling(data,labels):
    X = np.hstack((data, labels))
    if (len(X[X[:, -1] == 0])>len(X[X[:, -1]==1])):
        not_fewsamples = X[np.where(X[:, -1] == 0)]
        fewsamples = X[np.where(X[:, -1] == 1)]
    else:
        not_fewsamples = X[np.where(X[:, -1] == 1)]
        fewsamples = X[np.where(X[:, -1] == 0)]
    if len(fewsamples)==0:
        return data,labels
    print('np.shape(fewsamples)',np.shape(fewsamples))
    if (np.shape((np.unique(fewsamples,axis=0)))[0]<(len(not_fewsamples[:, -1])-len(fewsamples[:, -1]))):
        fewsamples_upsampled = resample(fewsamples,
                                        replace=True,  # sample with replacement
                                        n_samples=len(not_fewsamples[:, -1]) - len(fewsamples[:, -1]),
                                        # match number in majority class
                                        random_state=1)  # reproducible results
    else:
        fewsamples_upsampled = resample(fewsamples,
                                        replace=False,  # sample with replacement
                                        n_samples=len(not_fewsamples[:, -1])-len(fewsamples[:, -1]),  # match number in majority class
                                        random_state=1)  # reproducible results
    fewsamples_upsampled=np.vstack((fewsamples_upsampled, fewsamples))
    fewsamples_upsampled = np.vstack((fewsamples_upsampled, not_fewsamples))
    fewsamples_upsampled = shuffle(fewsamples_upsampled, random_state=42)
    labels = fewsamples_upsampled[:, -1]
    data = fewsamples_upsampled[:, 0:np.shape(fewsamples_upsampled)[1] - 1]

    return data,labels


def resampling(data, labels):
    AD_data = data[np.where(labels == 1)]
    AD_labels = labels[np.where(labels == 1)]
    Con_data = data[np.where(labels == 0)]
    Con_labels = labels[np.where(labels == 0)]
    indices = np.random.randint(0, len(AD_labels), len(AD_labels))
    AD_data = AD_data[indices].copy()
    AD_labels = AD_labels[indices].copy()
    indices = np.random.randint(0, len(Con_labels), len(Con_labels))
    Con_data = Con_data[indices].copy()
    Con_labels = Con_labels[indices].copy()
    data = concat(Con_data, AD_data)
    labels = concat(Con_labels[:, np.newaxis], AD_labels[:, np.newaxis])
    data, labels = shuffling(data, labels)
    return data, labels


def synthetic(data, labels, num):
    smote = ADASYN(ratio='all', n_neighbors=num)
    data, labels = smote.fit_sample(data, labels)

    return data, labels


def KSTest(train_data, test_data, step):
    index = []
    for i in range(0, len(train_data) - step, step):
        for j in range(train_data.shape[1]):

            r = ks_2samp(train_data[i:i + step, j], test_data[:, j])
            if r[1] > 0.05:
                index = np.append(index, j)
    if index==[]:
        return train_data,test_data            
    index = index[:, np.newaxis]
    index = index.astype(int)
    index = removeDuplicates(index)
    train_data[:, index] = 0
    test_data[:, index] = 0

    return train_data, test_data


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


def ica(data, number_of_combonents):
    ica = FastICA(n_components=number_of_combonents)
    ICA_combonents = ica.fit_transform(data)
    # ICA_combonents = ica.inverse_transform(ICA_combonents)

    return ICA_combonents


def pca(data, number_of_combonents):
    pca_m = PCA(n_components=number_of_combonents)
    PCA_combonents = pca_m.fit_transform(data)
    # ICA_combonents = pca_m.inverse_transform(ICA_combonents)

    return PCA_combonents


####3D transformation

def g_po_sk(input=None):
    input = input
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        sigma = 0.155
        input[:, :, :, i] = random_noise(input[:, :, :, i], var=sigma ** 2)
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='poisson')
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='speckle')
    input = np.moveaxis(input, 2, 1)
    return input


def sp(input):
    input = input
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='s&p')

    input = np.moveaxis(input, 2, 1)
    return input


def po(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='poisson')
    input = np.moveaxis(input, 2, 1)
    return input


def g_sp(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        sigma = 0.155
        input[:, :, :, i] = random_noise(input[:, :, :, i], var=sigma ** 2)
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='s&p')
    input = np.moveaxis(input, 2, 1)
    return input


def g_po(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        sigma = 0.155
        input[:, :, :, i] = random_noise(input[:, :, :, i], var=sigma ** 2)
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='poisson')
    input = np.moveaxis(input, 2, 1)
    return input


def g_sk(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        sigma = 0.155
        input[:, :, :, i] = random_noise(input[:, :, :, i], var=sigma ** 2)
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='speckle')
    input = np.moveaxis(input, 2, 1)
    return input


def sp_sk(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='s&p')
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='speckle')
    input = np.moveaxis(input, 2, 1)
    return input


def sp_po(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='s&p')
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='poisson')
    input = np.moveaxis(input, 2, 1)
    return input


def po_sk(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='poisson')
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='speckle')
    input = np.moveaxis(input, 2, 1)
    return input


def noise_all(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        sigma = 0.155
        input[:, :, :, i] = random_noise(input[:, :, :, i], var=sigma ** 2)
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='s&p')
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='poisson')
        input[:, :, :, i] = random_noise(input[:, :, :, i], mode='speckle')
    input = np.moveaxis(input, 2, 1)
    return input


def apply_noise_manytypes(data):
    data_noised = concatination(data, sp(data.copy()))
    data_noised = concatination(data_noised, g_po_sk(data.copy()))
    data_noised = concatination(data_noised, po(data.copy()))
    data_noised = concatination(data_noised, g_sp(data.copy()))
    data_noised = concatination(data_noised, g_po(data.copy()))
    data_noised = concatination(data_noised, g_sk(data.copy()))
    data_noised = concatination(data_noised, sp_sk(data.copy()))
    data_noised = concatination(data_noised, sp_po(data.copy()))
    data_noised = concatination(data_noised, po_sk(data.copy()))
    data_noised = concatination(data_noised, noise_all(data.copy()))
    return data_noised


def g_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def m_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def us_f(input):  # unsharp filter
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def c_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def w_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def g_m_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def g_us_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def m_us_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def g_c_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def g_w_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def c_w_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def m_w_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def m_c_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def m_g_c_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def m_g_us_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def c_w_us_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def c_w_us_g_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def c_w_us_m_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def all_f(input):
    shape = np.shape(input)
    num_of_inputs = shape[3]
    input = np.moveaxis(input, 1, 2)
    # print('input shape', np.shape(input))
    for i in range(num_of_inputs):
        input[:, :, :, i] = ndimage.gaussian_filter(input[:, :, :, i], .5)
        input[:, :, :, i] = nilearn.signal.clean(input[:, :, :, i])
        input[:, :, :, i] = wiener(input[:, :, :, i], mysize=7)
        input[:, :, :, i] = unsharp_mask(input[:, :, :, i], radius=5, amount=2)
        input[:, :, :, i] = signal.medfilt(input[:, :, :, i])
        # print(np.shape(input))
    input = np.moveaxis(input, 2, 1)
    return input


def apply_filter_manytypes(data):
    data_filter = g_f(data.copy())
    data_filter = concatination(data_filter, m_f(data.copy()))
    data_filter = concatination(data_filter, us_f(data.copy()))
    data_filter = concatination(data_filter, c_f(data.copy()))
    data_filter = concatination(data_filter, w_f(data.copy()))
    data_filter = concatination(data_filter, g_m_f(data.copy()))
    data_filter = concatination(data_filter, g_us_f(data.copy()))
    data_filter = concatination(data_filter, m_us_f(data.copy()))
    data_filter = concatination(data_filter, g_c_f(data.copy()))
    data_filter = concatination(data_filter, g_w_f(data.copy()))
    data_filter = concatination(data_filter, c_w_f(data.copy()))
    data_filter = concatination(data_filter, m_w_f((data.copy())))
    data_filter = concatination(data_filter, m_c_f(data.copy()))
    data_filter = concatination(data_filter, m_g_c_f(data.copy()))
    data_filter = concatination(data_filter, m_g_us_f(data.copy()))
    data_filter = concatination(data_filter, c_w_us_f(data.copy()))
    data_filter = concatination(data_filter, c_w_us_g_f(data.copy()))
    data_filter = concatination(data_filter, c_w_us_m_f(data.copy()))
    data_filter = concatination(data_filter, all_f(data.copy()))

    return data_filter


def concatination(data1, data2):
    shape_data1 = np.shape(data1)
    shape_data2 = np.shape(data2)
    matrix_data = np.zeros((shape_data1[0], shape_data1[1], shape_data1[2], shape_data1[3] + shape_data2[3]))
    matrix_data[:, :, :, 0:shape_data1[3]] = data1
    matrix_data[:, :, :, shape_data1[3]:shape_data1[3] + shape_data2[3]] = data2
    return matrix_data


def flatten(data):
    data = np.reshape(data, (np.shape(data)[0], -1))
    return data


def deflatten(data, shape):
    data = np.reshape(data, (-1, shape[1], shape[2], shape[3]))
    return data


def select_max_features(mask, number_of_featrues):
    mask_reduces = np.zeros((len(mask)))
    argsmask = np.argsort((-mask).copy())
    for i in range(number_of_featrues): mask_reduces[argsmask[i]] = mask[argsmask[i]]
    return mask_reduces


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
            resized_data = np.pad(data,
                                  ((0, 0), (round(diff), round(diff) - 1), (round(diff), round(diff) - 1), (0, 0)),
                                  'constant', constant_values=(0, 0))
        else:
            resized_data = np.pad(data, ((0, 0), (round(diff), round(diff)), (round(diff), round(diff)), (0, 0)),
                                  'constant', constant_values=(0, 0))

        return resized_data


def depth_reshapeing(data):
    depth = int(data.shape[3])

    dim0 = int(data.shape[0])

    dim1 = int(data.shape[1])

    dim2 = int(data.shape[2])

    step = math.floor(depth / 3)

    reshaped_data = np.empty((dim0, dim1, dim2, 3))

    for i in range(3):

        if i == 2:
            reshaped_data[:, :, :, i] = np.mean(data[:, :, :, step * i:depth], axis=3)
        else:
            reshaped_data[:, :, :, i] = np.mean(data[:, :, :, step * i:step * (i + 1)], axis=3)
    return reshaped_data


def converting_nii_to_npz(file_name):
    file_path = load_data.find_path(file_name)
    nii_file = data_augmentation.load_obj(file_path)
    np.savez(file_path[0:len(file_path) - 7] + '.npz', masked_voxels=nii_file)


def labels_convert_one_hot(labels):
    length = len(labels)
    if labels.all() == 0:
        ones = np.ones((length, 1))
        labels = np.hstack((ones, labels))
    elif labels.all() == 1:
        zeros = np.zeros((length, 1))
        labels = np.hstack((zeros, labels))

    else:
        zeros = np.zeros((length, 1))
        labels = np.hstack((zeros, labels))
        indecies = np.where(labels[:, 1] == 0)
        labels[indecies[0], 0] = 1
    return labels