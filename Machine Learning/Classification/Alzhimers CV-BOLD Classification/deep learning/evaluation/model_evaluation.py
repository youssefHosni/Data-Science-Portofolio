import matplotlib.pyplot as plt
import generate_result_
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def testing_and_printing(classification_model,classification_train,best_model,test_data,test_labels_one_hot,model_name,results_path,epoch):
    #classification_model.summary()
    test_eval = classification_model.evaluate(test_data, test_labels_one_hot, verbose=1)
    prediction=classification_model.predict(test_data)
    predicted_classes=classification_model.predict_classes(test_data)
    print(predicted_classes)
    test_labels=test_labels_one_hot[:,1]
    
    print(test_labels_one_hot)
    print(prediction)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    precision = precision_score(test_labels, predicted_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(test_labels, predicted_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(test_labels, predicted_classes)
    print('F1 score: %f' % f1)
    
    matrix = confusion_matrix(test_labels, predicted_classes)
    print(matrix)
    
        
    
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

    
    test_acc=best_model.evaluate(test_data,test_labels_one_hot)
    print('best model test accaurecy is ',test_acc)
    generate_result_.cnn_save_result(test_eval[1],classification_model,model_name,results_path)
