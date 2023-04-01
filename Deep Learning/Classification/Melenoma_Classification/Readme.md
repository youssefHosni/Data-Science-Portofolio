# Melenoma Classification

## 1. Problem Statment

Classfying Melenoma skin lesion images into 9 classes of diagnostics using deep learning models.  

## 2. Methods

### 2.1. Dataset
The dataset used is the [Skin Lesion Images for Melanoma Classification on Kaggle](https://www.kaggle.com/datasets/andrewmvd/isic-2019). This dataset contains the training data for the ISIC 2019 challenge, note that it already includes data from previous years (2018 and 2017). The dataset for ISIC 2019 contains 25,331 images available for the classification of dermoscopic images among nine different diagnostic categories:

* Melanoma
* Melanocytic nevus
* Basal cell carcinoma
* Actinic keratosis
* Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
* Dermatofibroma
* Vascular lesion
* Squamous cell carcinoma
* None of the above

![Screenshot 2023-03-30 183038](https://user-images.githubusercontent.com/72076328/228887596-f9be3bed-ad19-4469-8fda-09a8725bb246.png)

### 2.2. Data Preprcoessing 


### 2.3. Feature Engineering 


### 2.4. Models
Finetuned pretrained CNN based models. The models used are: 

* VGG-16
* VGG-19
* ResNet-50 
* Mobile-Net


## 3. Results 
Since the data is imbalanced so the F1_score.

