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
import os
import CNN_feature_extractor
import model_evaluation
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16


def model (train_data_whole,train_labels_whole,test_data,test_labels,opt,epoch,batch_size_factor,num_classes,result_path,feature_extraction,feature_extractor_parameters):
    
    train_data, valid_data, train_labels, valid_labels = train_test_split(train_data_whole, train_labels_whole,test_size=0.1, random_state=13)
    train_labels_one_hot=data_preprocessing.labels_convert_one_hot(train_labels)
    valid_labels_one_hot=data_preprocessing.labels_convert_one_hot(valid_labels)
    test_labels_one_hot=data_preprocessing.labels_convert_one_hot(test_labels)    
    batch_size=round(train_data.shape[0]/batch_size_factor)
    
    input_shape= (224,224,3)
    vgg_model = VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=input_shape)
    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block2_pool'].output
    
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    
    # Creating new model. Please note that this is NOT a Sequential() model.
    classification_model = Model(input=vgg_model.input, output=x)
    for layer in classification_model.layers[:7]:
        layer.trainable = True
    
    es=keras.callbacks.EarlyStopping(monitor='val_acc',
                                  min_delta=0,
                                  patience=5000,
                                  verbose=1, mode='auto')

    mc = ModelCheckpoint(os.path.join(result_path,'best_model.h5'), monitor='val_acc', mode='auto', save_best_only=True)
    
    classification_model.compile(loss='mean_squared_error',optimizer=opt,metrics=['accuracy'])
    classification_train = classification_model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epoch,
                                                    verbose=1, validation_data=(valid_data, valid_labels_one_hot),callbacks=[mc,es])
    best_model=load_model(os.path.join(result_path,'best_model.h5'))
    file_name=os.path.split(result_path)[1]    
    date=os.path.split(os.path.split(result_path)[0])[1]
    classification_model.save(os.path.join(result_path,date+'_'+file_name+'_'+'VGGpretrained_model.h5')) 

    if feature_extraction==1:
        feature_extractor_parameters['CNN_model']=classification_model
        CNN_feature_extractor.CNN_feature_extraction_classsification(feature_extractor_parameters,result_path)
        return    
    model_evaluation.testing_and_printing(classification_model,classification_train,best_model,test_data, test_labels_one_hot, 'VGG_pretrained', result_path,epoch)
