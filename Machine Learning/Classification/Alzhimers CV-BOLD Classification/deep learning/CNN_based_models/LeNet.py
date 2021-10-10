import data_preprocessing
import keras
from keras.models import Sequential
from keras.layers import  Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import  layers
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LeakyReLU
import os
import CNN_feature_extractor
import model_evaluation
from sklearn.model_selection import train_test_split

def model (train_data_whole,train_labels_whole,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters):
    
    train_data, valid_data, train_labels, valid_labels = train_test_split(train_data_whole, train_labels_whole,test_size=0.1, random_state=13)
    train_labels_one_hot=data_preprocessing.labels_convert_one_hot(train_labels)
    valid_labels_one_hot=data_preprocessing.labels_convert_one_hot(valid_labels)
    test_labels_one_hot=data_preprocessing.labels_convert_one_hot(test_labels)
    batch_size=round(train_data.shape[0]/batch_size_factor)
    classification_model = Sequential()
    # C1 Convolutional Layer
    classification_model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1),activation='relu', input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]), padding='same'))
    
    classification_model.add(Dropout(0.7))

    # S2 Pooling Layer
    classification_model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    # C3 Convolutional Layer
    classification_model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid',activation='relu'))
  
    classification_model.add(Dropout(0.7))

    # S4 Pooling Layer
    classification_model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # C5 Fully Connected Convolutional Layer
    classification_model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), padding='valid',activation='relu'))
   
    classification_model.add(Dropout(0.8))
    #Flatten the CNN output so that we can connect it with fully connected layers
    classification_model.add(layers.Flatten())
    # FC6 Fully Connected Layer
    classification_model.add(layers.Dense(84,activation='relu'))
    
    classification_model.add(Dropout(0.8))

    #Output Layer with softmax activation
    classification_model.add(layers.Dense(num_classes, activation='softmax'))


    classification_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])


    es=keras.callbacks.EarlyStopping(monitor='val_acc',
                                  min_delta=0,
                                  patience=100,
                                  verbose=1, mode='auto')
    mc = ModelCheckpoint(os.path.join(result_path,'best_model.h5'), monitor='val_acc', mode='auto', save_best_only=True)
    classification_train = classification_model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epoch,
                                                    verbose=1, validation_data=(valid_data, valid_labels_one_hot),callbacks=[es,mc])
    best_model=load_model(os.path.join(result_path,'best_model.h5'))
    file_name=os.path.split(result_path)[1]    
    date=os.path.split(os.path.split(result_path)[0])[1]
    classification_model.save(os.path.join(result_path,date+'_'+file_name+'_'+'leNet_model.h5')) 
    if feature_extraction==1:
        feature_extractor_parameters['CNN_model']=classification_model
        CNN_feature_extractor.CNN_feature_extraction_classsification(train_data_whole,train_labels_whole,test_data,test_labels,feature_extractor_parameters,result_path)
        return 
    model_evaluation.testing_and_printing(classification_model,classification_train,best_model,test_data,test_labels_one_hot,'LeNet',result_path,epoch)    