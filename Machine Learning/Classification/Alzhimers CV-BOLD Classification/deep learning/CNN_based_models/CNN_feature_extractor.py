__version__ = '0.10.3'

import matplotlib.pyplot as plt
import tensorflow as tf
import load_data
import optimizers
import data_preprocessing
import keras
from keras import  layers
from keras import losses
from keras import backend as K
from keras.models import Sequential, Input, Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from itertools import islice
import sys
import os

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def fully_connected_layer(dropouts,activation_function):
   
    if dropouts==None or activation_function==None:
        return
         
    
    classification_model = Sequential()
    # 1st Fully Connected Layer
    classification_model.add(Dense(4096,activation=activation_function[0]))
    # Add Dropout to prevent overfitting
    classification_model.add(Dropout(dropouts[0]))
    # 2nd Fully Connected Layer
    classification_model.add(Dense(4096,activation=activation_function[1]))
    # Add Dropout
    classification_model.add(Dropout(dropouts[1]))
    # 3rd Fully Connected Layer
    classification_model.add(Dense(1000,activation=activation_function[2]))
    # Add Dropout
    classification_model.add(Dropout(dropouts[2]))
    # Output Layer
    classification_model.add(Dense(2,activation=activation_function[3]))  
    return classification_model

def fully_connected_layer_fit(features_train,features_test,train_labels,test_labels,classification_model,epoch,opt):    
    train_labels=data_preprocessing.labels_convert_one_hot(train_labels)
    test_labels=data_preprocessing.labels_convert_one_hot(test_labels)
    features_train, features_valid, train_labels, valid_labels = train_test_split(features_train, train_labels,test_size=0.1, random_state=13)
    print(features_train.shape)
    print(features_valid.shape)
    batch_size=features_train.shape[0]  
    classification_model.compile(loss=losses.binary_crossentropy, optimizer=opt,
                             metrics=['accuracy'])
    classification_train = classification_model.fit(features_train, train_labels, batch_size=batch_size, epochs=epoch,
                                            verbose=1, validation_data=(features_valid, valid_labels))
    test_eval = classification_model.evaluate(features_test, test_labels, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    #print('AUC on test data:',test_eval[2])
    print('the number of epochs:', epoch)
    accuracy = classification_train.history['acc']
    val_accuracy = classification_train.history['val_acc']
    loss = classification_train.history['loss']
    val_loss = classification_train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    return test_eval


def features_pretrained_model(model,train_data,test_data):
    features_train=model.predict(train_data)
    features_test=model.predict(test_data)
    length=features_train.shape[1]
    width=features_train.shape[2]
    depth=features_train.shape[3]
    features_train=features_train.reshape(features_train.shape[0],length*width*depth)
    features_test=features_test.reshape(features_test.shape[0],length*width*depth)
    return features_train,features_test

def features_saved_model(train_data,test_data,model,intermediate_layer,results):
    
    if type(model)==str:
        results['model_is']=model
        model_path=load_data.find_path(model)
        model=load_model(model_path)
        
    
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[intermediate_layer].output])
     # output in test mode = 0
    features_train = get_layer_output([train_data, 0])[0]
    
    # output in train mode = 1
    features_test = get_layer_output([test_data, 1])[0]
    
    length=features_train.shape[1]
    width=features_train.shape[2]
    depth=features_train.shape[3]
    features_train=features_train.reshape(features_train.shape[0],length*width*depth)
    features_test=features_test.reshape(features_test.shape[0],length*width*depth)
    
    return features_train,features_test,results
    
  
def classifer_fit_testing(features_train,train_labels,features_test,test_labels,classifer_name,hyperparameters,results):
    opt=optimizers.choosing(hyperparameters['opt'])
    kernel = 1.0 * RBF(features_train.shape[1])
    
    classifers = {'SVC': SVC(gamma='scale'),
                  'NuSVC': NuSVC(probability=True, gamma='scale', class_weight='balanced'),
                  'LinearSVC': LinearSVC(random_state=0, tol=1e-5, penalty=hyperparameters['penalty'], dual=False, C=hyperparameters['C']),
                  'KNN': KNeighborsClassifier(hyperparameters['neighbors'], weights='distance', p=2, leaf_size=100),
                  'Random_forest': RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
                  'Decision_tree': DecisionTreeClassifier(),
                  'GaussinanNB': GaussianNB(),
                  'Adaboost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1),
                  'Gradientboos': GradientBoostingClassifier(),
                  'Gradient_boosting': GradientBoostingClassifier(),
                  'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
                  'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
                  'gpc':GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1),
                  'fully_connected':fully_connected_layer(hyperparameters['dropouts'],hyperparameters['activation_function'])}
        
    if type(classifer_name)==str and classifer_name !='all' :
         classifer=classifers[classifer_name]
         print(classifer_name)
         if classifer_name=='fully_connected':
             test_acc=fully_connected_layer_fit(features_train,features_test,train_labels,test_labels,classifer,hyperparameters['epoch'],opt)     
             results['classifer_is']=classifer_name
             results['test_acc_is']=test_acc 
         else:
             classifer.fit(features_train,train_labels)
             predicted_labels=classifer.predict(features_test)
             test_acc=accuracy_score(test_labels, predicted_labels)
             results['classifer_is']=classifer_name
             results['test_acc_is']=test_acc 
    else:
      if classifer_name=='all':
          #classifer=take(len(classifers)-2, classifers.items())
          classifer=classifers
      else:
         classifer=[classifers.get(clf_name) for clf_name in classifer_name]
      i=0   
      for classifer_name in classifer:
         print(classifer_name)
         clf=classifer[classifer_name]
         if classifer_name=='fully_connected':
             test_acc=fully_connected_layer_fit(features_train,features_test,train_labels,test_labels,clf,hyperparameters['epoch'],opt)     
             test_acc=accuracy_score(test_labels, predicted_labels)
             results['classifer_'+str(i)+'is']=classifer_name
             results['test'+str(i)+'acc_is']=test_acc      
         else:
             clf.fit(features_train,train_labels)
             predicted_labels=clf.predict(features_test)
             test_acc=accuracy_score(test_labels, predicted_labels)
             results['classifer_'+str(i)+'is']=classifer_name
             results['test'+str(i)+'acc_is']=test_acc 
         i=i+1
    return results

def CNN_feature_extraction_classsification(feature_extractor_parameters,results_path):
    '''
    function description
    the function use the CNN as feature extractor and use this features to train
    and test number of classifers 
    
    Function arguments :
        train_data: the training data from which the training features will be extracted
        test_data :the training data from which the training features will be extracted
        train_labels:the labels for the training data in the normal form
        test_labels:the labels for the training data in the normal form
        model_type: there are two values for this variable 'pretrained_model' or
        'saved_model', the first the pretrained model provided by keras will be used , 
        and the second one of the saved models will be used and this will be defined by the 
        comming variable
        feature_extractor_parameters: 
        This is a dict of four elemnts as the follwoing:
            CNN_model: this variable defines which model to be used , if it is value is 'all'
            this will mean that all the pretrained models will be used , this can be used only if 
            "model_type" is 'pretrained_model', and if the "model_type" is 'pretraiend_model' then
            the value should be one of the pretrained models else it will give error, and it will take 
            the model name as it is saved if the "model_type" is 'saved_model'.
            classifer_name: this variable is used to choose the classifer to be used to be 
            trained by the training data and to be tested by the test_data , it can be equal to 
            'all'm which will mean that all the classifers will be used or it can take the name of one
            of the classifers,the name should match the names defined in the function else there be error.
            hyperprametres: this is another dict in which the classifer hyperparmaters is defined, it contain four main
            parameters , the value of the neigbors to look to for the KNN classifer and the for SVC classifer thera are
            two hyperparamters 'penalty' and 'C' and for the fully conncted classifer there are two hyperparamters
            'dropouts','activation_functions','epoch'and 'opt'.If any of this classifers is not used then it's hyperparmater should 
            be =None.
        results_path:this will be the path where the results file will be generated and the results will be printed.   
        
    '''
    
    CNN_model=feature_extractor_parameters['CNN_model']
    model_type=feature_extractor_parameters['model_type']
    classifer_name=feature_extractor_parameters['classifer_name']
    intermediate_layer=feature_extractor_parameters['intermediate_layer']
    hyperparameters=feature_extractor_parameters['hyperparameters']
    data=feature_extractor_parameters['data']
    train_data=data['train_data']
    test_data=data['test_data']
    train_labels=data['train_labels']
    test_labels=data['test_labels']
    if model_type=='pretrained':
        
        if CNN_model=='all':
            model={'Xcception':keras.applications.xception.Xception(include_top=False, weights='imagenet'),
                    'VGG16':VGG16(weights='imagenet',include_top=False),
                    'VGG19':VGG19(include_top=False, weights='imagenet'),              
                    'ResNet50':keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet'),
                    'inceptionv2':keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet'),                        
                    'inceptionv3':InceptionV3(include_top=False, weights='imagenet'), 
                    'DensNet121':keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet'),
                    'DenseNeT169':DenseNet169(include_top=False, weights='imagenet'),
                    'DensNet201':DenseNet201(include_top=False, weights='imagenet'),
                    'MobileNet':MobileNet( alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet'),
                    'NASNet':NASNetMobile( include_top=False, weights='imagenet'),
                    'MobileNetV2':MobileNetV2( alpha=1.0,  include_top=False, weights='imagenet')}
        
            for model_name in model:
                print(model_name)
                results={}
                results['the_model_is'] =model_name
                features_train,features_test=features_pretrained_model(model[model_name],train_data,test_data)
                results=classifer_fit_testing(features_train,train_labels,features_test,test_labels,classifer_name,hyperparameters,results)
                for key in results:
                    f=open(os.path.join(results_path,'CNN_as_feature_extraction_results.txt'),"a+")
                    line1=[key,results[key]]
                    print(line1)
                    f.write("{}" "\n" .format(line1)) 

            
        else:
            if CNN_model=='Xcception':
                model=keras.applications.xception.Xception(include_top=False, weights='imagenet')
            elif CNN_model=='VGG16':
                model= VGG16(weights='imagenet',include_top=False)
            elif CNN_model=='VGG19':
                model=VGG19(weights='imagenet',include_top=False)
            elif CNN_model=='ResNet50':
                model=keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
            elif CNN_model=='inceptionV2':
                model=keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet')
            elif CNN_model=='inceptionV3':
                model=InceptionV3(include_top=False, weights='imagenet')
            elif CNN_model=='DenseNet121':
                model=keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet')
            elif CNN_model=='DenseNet169':
                model=DenseNet169(include_top=False, weights='imagenet')
            elif CNN_model=='DenseNet201':
                model=DenseNet201(include_top=False, weights='imagenet')
            elif CNN_model=='MobilNet':
                model=MobileNet(include_top=False, weights='imagenet')
            elif CNN_model=='NASNet':
                model=NASNetMobile(include_top=False, weights='imagenet')    
            elif CNN_model=='MobileNetV2':
                model=MobileNetV2(include_top=False, weights='imagenet') 
            else:
                print('No such a pretrained model with such a name')
                sys.exit()
           
            model_name=CNN_model
            print(model_name)
            results={}
            results['the_model_is'] =model_name
            features_train,features_test=features_pretrained_model(model,train_data,test_data)
            results=classifer_fit_testing(features_train,train_labels,features_test,test_labels,classifer_name,hyperparameters,results)
            for key in results:
                f=open(os.path.join(results_path,'CNN_as_feature_extraction_results.txt'),"a+")
                line1=[key,results[key]]
                print(line1)
                f.write("{}" "\n" .format(line1)) 

                
    
    elif model_type=='saved_model':
        results={}
        features_train,featuers_test,results=features_saved_model(train_data,test_data,CNN_model,intermediate_layer,results)
        results=classifer_fit_testing(features_train,train_labels,featuers_test,test_labels,classifer_name,hyperparameters,results)
        
        for key in results:
            f=open(os.path.join(results_path,'CNN_as_feature_extraction_results.txt'),"a+")
            line1=[key,results[key]]
            print(line1)
            f.write("{}" "\n" .format(line1)) 

    
    f=open(os.path.join(results_path,'classifers_hyperparameters.txt'),"w+")
    line1='the classifers hyperparameters used '
    line2=hyperparameters
    f.write("{}" "\n" "{}" "\n" .format(line1,line2))
    
        
    
    return 
