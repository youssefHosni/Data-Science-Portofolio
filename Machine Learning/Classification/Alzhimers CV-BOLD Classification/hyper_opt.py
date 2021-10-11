import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from data_preprocessing import select_max_features


def hyperparameter_selection(data,labels,number_of_cv,feature_selection_type,Hyperparameter,mask_threshold):
    feature_weight=np.zeros(np.shape(data)[1])
    weights = np.zeros(np.shape(data)[1])
    if (feature_selection_type=='recursion'):
         masks = []
         accuracies = []
         weights = []
         if Hyperparameter is not None:
             #boundaries between 1e+4 to 1e+10
             Hyperparameter=[Hyperparameter]
         else:
             Hyperparameter=[500,1000,2000,3000,1e+4,5000,1e+6,7000,1e+8,9000,1e+10]
         for i, value in enumerate(Hyperparameter):
              svc = SVC(kernel="linear")
              rfe = RFE(estimator=svc, step=1,n_features_to_select=value)
              rfe = rfe.fit(data, labels)
              lsvc = cross_validate(rfe.estimator_, data, labels, cv=number_of_cv, scoring='accuracy', return_estimator=True)
              index_of_max_accuracy = np.argmax(lsvc['test_score'])
              accuracy = lsvc['test_score'][index_of_max_accuracy]
              weight = np.absolute(lsvc['estimator'][index_of_max_accuracy].coef_)
              for i, estimator in enumerate(lsvc['estimator']):
                   model = SelectFromModel(estimator, prefit=True, threshold="mean")
                   indecies = model.get_support(indices=True)
                   T_new = model.transform(data)
                   nfeatures = T_new.shape[1]
                   feature_weight[indecies] = feature_weight[indecies] + 1
              mask = np.array(feature_weight > mask_threshold, dtype=int)[np.newaxis, :]
              masks = np.reshape(masks, (-1, np.shape(mask)[1]))
              masks = np.append(masks, mask, axis=0)
              accuracies = np.append(accuracies, accuracy)
              weights = np.append(weights, weight)
              weights = np.reshape(weights, (-1, np.shape(mask)[1]))
         argument_of_maximum_accuracy = np.argmax(accuracies)
         return masks[argument_of_maximum_accuracy], accuracies[argument_of_maximum_accuracy], weights[argument_of_maximum_accuracy]
    if (feature_selection_type=='L2_penality'):
         masks=[]
         accuracies=[]
         weights=[]
         #model_threshold=.00075
         if Hyperparameter is not None:
             #boundaries between 1e-4 to 10000
             Hyperparameter=[Hyperparameter]
         else:
             Hyperparameter=[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]
         for i,value in enumerate(Hyperparameter):
              lsvc = LinearSVC(C=value, penalty="l2", dual=True,max_iter=40000)
              lsvc = cross_validate(lsvc, data,labels, cv=number_of_cv, scoring = 'accuracy', return_estimator =True)
              index_of_max_accuracy=np.argmax(lsvc['test_score'])
              accuracy=lsvc['test_score'][index_of_max_accuracy]
              weight= np.absolute(lsvc['estimator'][index_of_max_accuracy].coef_)
              for i,estimator in enumerate(lsvc['estimator']):
                   model = SelectFromModel(estimator, prefit=True,threshold='1.25*median')
                   indecies=model.get_support(indices=True)
                   T_new = model.transform(data)
                   nfeatures=T_new.shape[1]
                   feature_weight[indecies]=feature_weight[indecies]+1

              mask=np.array(feature_weight>mask_threshold,dtype=int)[np.newaxis, :]
              masks=np.reshape(masks,(-1,np.shape(mask)[1]))
              masks=np.append(masks,mask,axis = 0)
              accuracies=np.append(accuracies,accuracy)
              weights=np.append(weights,weight)
              weights=np.reshape(weights,(-1,np.shape(mask)[1]))
         argument_of_maximum_accuracy=np.argmax(accuracies)
         return masks[argument_of_maximum_accuracy],accuracies[argument_of_maximum_accuracy],weights[argument_of_maximum_accuracy]

def model(data_train,labels_train,mask,data_validation=None,labels_validation=None,model_type='gaussian_process'):
    if (model_type=='gaussian_process'):
         kernel = 1.0 * RBF(len(mask[mask>0])*40)
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc=gpc.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
         if (data_validation is not None):
              validation_accuracy=gpc.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
              return gpc,validation_accuracy,model_type
         return gpc,model_type
    if (model_type=='svm'):
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf=clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
         if (data_validation is not None):
              validation_accuracy=clf.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
              return clf,validation_accuracy,model_type
         return clf,model_type
    if (model_type=='decison tree classifer'):
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
         if (data_validation is not None):
              validation_accuracy=tree_clf.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
              return tree_clf,validation_accuracy,model_type
         return tree_clf,model_type
    if (model_type=='ensamble classifer'):
         kernel = 1.0 * RBF(len(mask[mask>0]))
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc=gpc.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf=clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)

         estimators=[('Gussian_process',gpc),('svm classifer',clf),('Decision tree',tree_clf)]
         ensemble = VotingClassifier(estimators, voting='hard',)
         ensemble=ensemble.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
         if (data_validation is not None):
              validation_accuracy=ensemble.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
              return ensemble,validation_accuracy,model_type
         return ensemble,model_type


def create_mask(data,labels,number_of_cv,feature_selection_type,Hyperparameter,mask_threshold=1,model_type='gaussian_process'):
     index=0
     masks=np.zeros(np.shape(data)[1])[np.newaxis, :]
     accuracies=np.zeros((number_of_cv+1,1))
     wight_matrix=np.zeros(np.shape(data)[1])[np.newaxis, :]
     weights=np.zeros(np.shape(data)[1])
     if Hyperparameter is not None:
          # boundaries between 1e-4 to 10000
          Hyperparameter = Hyperparameter
     else:
          Hyperparameter = np.shape(data)[1]
     kf = KFold(n_splits=number_of_cv)
     for train_index, test_index in kf.split(data):
          X_train, X_test = data[train_index], data[test_index]
          y_train, y_test = labels[train_index], labels[test_index]
          mask,accuracy,weights=hyperparameter_selection(X_train,y_train,number_of_cv,feature_selection_type,Hyperparameter,mask_threshold)
          if (len(mask[mask>0])==0):
               continue
          model_,validation_accuracy,model_type=model(X_train,y_train,mask,X_test,y_test,model_type)
          masks=np.append(masks,mask[np.newaxis, :], axis=0)
          accuracies[index]=validation_accuracy
          wight_matrix=np.append(wight_matrix,weights[np.newaxis, :],axis=0)
          index=index+1
     optimal_mask = select_max_features(np.sum(masks,axis=0).copy(), Hyperparameter)
     optimal_mask=np.array(optimal_mask>mask_threshold,dtype=int)
     if (len(optimal_mask[optimal_mask>0])==0):
          raise ValueError("the mask produces zero array")
     out_model,model_type=model(data,labels,optimal_mask,None,None,model_type)
     argument_of_maximum_accuracy=np.argmax(accuracies)
     return optimal_mask,out_model,model_type,wight_matrix[argument_of_maximum_accuracy]


def model_1D(data_train,labels_train,mask,data_validation=None,labels_validation=None,model_type='gaussian_process'):
    data_train=np.reshape(data_train,(-1,1))
    if (model_type=='gaussian_process'):
         kernel = 1.0 * RBF(len(mask[mask>0]))
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc=gpc.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=gpc.score(data_validation,labels_validation)
              return gpc,validation_accuracy,model_type
         return gpc,model_type
    if (model_type=='svm'):
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf=clf.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=clf.score(data_validation,labels_validation)
              return clf,validation_accuracy,model_type
         return clf,model_type
    if (model_type=='decison tree classifer'):
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=tree_clf.score(data_validation,labels_validation)
              return tree_clf,validation_accuracy,model_type
         return tree_clf,model_type
    if (model_type=='ensamble classifer'):
         kernel = 1.0 * RBF(len(mask[mask>0]))
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc=gpc.fit(data_train,labels_train)
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf=clf.fit(data_train,labels_train)
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train,labels_train)

         estimators=[('Gussian_process',gpc),('svm classifer',clf),('Decision tree',tree_clf)]
         ensemble = VotingClassifier(estimators, voting='hard',)
         ensemble=ensemble.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=ensemble.score(data_validation,labels_validation)
              return ensemble,validation_accuracy,model_type
         return ensemble,model_type
def model_1D_calibrate(data_train,labels_train,mask,data_validation=None,labels_validation=None,model_type='gaussian_process'):
    data_train=np.reshape(data_train,(-1,1))
    if (model_type=='gaussian_process'):
         kernel = 1.0 * RBF(len(mask[mask>0])**2)
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc = CalibratedClassifierCV(gpc, cv=5, method='isotonic')
         gpc=gpc.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=gpc.score(data_validation,labels_validation)
              return gpc,validation_accuracy,model_type
         return gpc,model_type
    if (model_type=='svm'):
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf = CalibratedClassifierCV(clf, cv=5, method='isotonic')
         clf=clf.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=clf.score(data_validation,labels_validation)
              return clf,validation_accuracy,model_type
         return clf,model_type
    if (model_type=='decison tree classifer'):
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=tree_clf.score(data_validation,labels_validation)
              return tree_clf,validation_accuracy,model_type
         return tree_clf,model_type
    if (model_type=='ensamble classifer'):
         kernel = 1.0 * RBF(len(mask[mask>0]))
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc=gpc.fit(data_train,labels_train)
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf=clf.fit(data_train,labels_train)
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train,labels_train)

         estimators=[('Gussian_process',gpc),('svm classifer',clf),('Decision tree',tree_clf)]
         ensemble = VotingClassifier(estimators, voting='hard',)
         ensemble=ensemble.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=ensemble.score(data_validation,labels_validation)
              return ensemble,validation_accuracy,model_type
         return ensemble,model_type
def model_reduced(data_train,labels_train,mask,data_validation=None,labels_validation=None,model_type='gaussian_process'):
    if (model_type=='gaussian_process'):
         kernel = 1.0 * RBF(1)
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc=gpc.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=gpc.score(data_validation,labels_validation)
              return gpc,validation_accuracy,model_type
         return gpc,model_type
    if (model_type=='svm'):
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf=clf.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=clf.score(data_validation,labels_validation)
              return clf,validation_accuracy,model_type
         return clf,model_type
    if (model_type=='decison tree classifer'):
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=tree_clf.score(data_validation,labels_validation)
              return tree_clf,validation_accuracy,model_type
         return tree_clf,model_type
    if (model_type=='ensamble classifer'):
         kernel = 1.0 * RBF(len(mask[mask>0]))
         gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                          multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
         gpc=gpc.fit(data_train,labels_train)
         clf = svm.SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
         clf=clf.fit(data_train,labels_train)
         tree_clf = tree.DecisionTreeClassifier()
         tree_clf = tree_clf.fit(data_train,labels_train)

         estimators=[('Gussian_process',gpc),('svm classifer',clf),('Decision tree',tree_clf)]
         ensemble = VotingClassifier(estimators, voting='hard',)
         ensemble=ensemble.fit(data_train,labels_train)
         if (data_validation is not None):
              validation_accuracy=ensemble.score(data_validation,labels_validation)
              return ensemble,validation_accuracy,model_type
         return ensemble,model_type
