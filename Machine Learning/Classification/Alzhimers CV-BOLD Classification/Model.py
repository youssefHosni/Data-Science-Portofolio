import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from numpy import load
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
def inner_loop(data,labels,number_of_cv,feature_selection_type,Hyperparameter,mask_threshold):
     feature_weight=np.zeros(np.shape(data)[1])
     weights = np.zeros(np.shape(data)[1])
     if (feature_selection_type=='recursion'):
          svc = SVC(kernel="linear")
          rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(number_of_cv),
                      scoring='accuracy',n_jobs=-1,min_features_to_select=Hyperparameter)
          rfecv=rfecv.fit(data,labels)
          index_of_max_accuracy=np.argmax(rfecv.grid_scores_)
          accuracy=rfecv.grid_scores_[index_of_max_accuracy]
          #print("Optimal number of features : %d" % rfecv.n_features_)
          indecies=rfecv.get_support( indices=True)
          weights[indecies]= np.absolute(rfecv.estimator_.coef_)
          weights=weights[np.newaxis, :]
          feature_weight[indecies]=np.array(feature_weight[indecies]+1,dtype=int)
          mask=feature_weight
          return mask,accuracy,weights
     if (feature_selection_type=='L2_penality'):
          #model_threshold=.000005
          lsvc = LinearSVC(C=Hyperparameter, penalty="l2", dual=True,max_iter=40000)
          lsvc = cross_validate(lsvc, data,labels, cv=number_of_cv, scoring = 'accuracy', return_estimator =True)
          index_of_max_accuracy=np.argmax(lsvc['test_score'])
          accuracy=lsvc['test_score'][index_of_max_accuracy]
          weights= np.absolute(lsvc['estimator'][index_of_max_accuracy].coef_)
          for i,estimator in enumerate(lsvc['estimator']):
               model = SelectFromModel(estimator, prefit=True,threshold="mean")
               indecies=model.get_support(indices=True)
               T_new = model.transform(data)
               nfeatures=T_new.shape[1]
               feature_weight[indecies]=feature_weight[indecies]+1
          mask=np.array(feature_weight>mask_threshold,dtype=int)
          #print('Optimal number of features:',np.shape(mask[mask>0]))
          return mask,accuracy,weights


def model(data_train,labels_train,data_validation,labels_validation,mask,model_type,sample_weights):
     if (model_type=='gaussian_process'):
          kernel = 1.0 * RBF(len(mask[mask>0])+1)
          gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                           multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
          gpc=gpc.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
          validation_accuracy=gpc.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
          return gpc,validation_accuracy,model_type
     if (model_type=='svm'):
          clf = SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
          clf=clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train,sample_weight=sample_weights)
          validation_accuracy=clf.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
          return clf,validation_accuracy,model_type
     if (model_type=='decison tree classifer'):
          tree_clf = tree.DecisionTreeClassifier()
          tree_clf = tree_clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
          validation_accuracy=tree_clf.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
          return tree_clf,validation_accuracy,model_type
     if (model_type=='ensamble classifer'):
          kernel = 1.0 * RBF(len(mask[mask>0]))
          gpc = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5,random_state=None,
                                           multi_class="one_vs_rest",max_iter_predict=100,n_jobs=-1)
          gpc=gpc.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
          clf = SVC(kernel='linear',gamma='scale', decision_function_shape='ovo')
          clf=clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
          tree_clf = tree.DecisionTreeClassifier()
          tree_clf = tree_clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
          estimators=[('Gussian_process',gpc),('svm classifer',clf),('Decision tree',tree_clf)]
          ensemble = VotingClassifier(estimators, voting='hard',)
          ensemble=ensemble.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train)
          validation_accuracy=ensemble.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)
          return ensemble,validation_accuracy,model_type
     if (model_type=='Random_forest'):
          clf=RandomForestClassifier(n_jobs=-1,max_depth=len(mask[mask>0]))
          clf=clf.fit(data_train[:,np.squeeze(np.where(mask>0),axis=0)],labels_train,sample_weight=sample_weights)
          validation_accuracy=clf.score(data_validation[:,np.squeeze(np.where(mask>0),axis=0)],labels_validation)  
          return clf,validation_accuracy,model_type   
      


def create_mask(data,labels,number_of_cv,feature_selection_type,Hyperparameter,mask_threshold,model_type,sample_weights):
     index=0
     models=[]
     masks=np.zeros(np.shape(data)[1])[np.newaxis, :]
     accuracies=np.zeros((number_of_cv+1,1))
     wight_matrix=np.zeros(np.shape(data)[1])[np.newaxis, :]
     weights=np.zeros(np.shape(data)[1])
     kf = KFold(n_splits=number_of_cv)
     for train_index, test_index in kf.split(data):
          X_train, X_test = data[train_index], data[test_index]
          y_train, y_test = labels[train_index], labels[test_index]
          sample_weights_new=sample_weights[train_index]
          mask,accuracy,weights=inner_loop(X_train,y_train,number_of_cv,feature_selection_type,Hyperparameter,mask_threshold)
          if (len(mask[mask>0])==0):
               continue
          model_,validation_accuracy,model_type=model(X_train,y_train,X_test,y_test,mask,model_type,sample_weights_new)
          masks=np.append(masks,mask[np.newaxis, :], axis=0)
          print(masks.shape)
          accuracies[index]=validation_accuracy
          print(validation_accuracy)
          print(np.shape(np.where(mask>0)))
          wight_matrix=np.append(wight_matrix,weights,axis=0)
          models=np.append(models,model_)
          index=index+1
     argument_of_maximum_accuracy=np.argmax(accuracies)
     print(argument_of_maximum_accuracy)
     print('total number of features:',np.shape(masks[argument_of_maximum_accuracy][masks[argument_of_maximum_accuracy]>0]))
     if (len(masks[argument_of_maximum_accuracy][masks[argument_of_maximum_accuracy]>0])==0):
          raise ValueError("the mask produces zero array")

     return masks[argument_of_maximum_accuracy+1],models[argument_of_maximum_accuracy],model_type,wight_matrix[argument_of_maximum_accuracy+1]


'''
#unit test
oulu_con_data=load('/data/fmri/Folder/AD_classification/Data/input_data/whole_brain_Oulu_Con.npz')['masked_voxels']
oulu_ad_data=load('/data/fmri/Folder/AD_classification/Data/input_data/whole_brain_Oulu_AD.npz')['masked_voxels']
lst_oulu=np.hstack((oulu_ad_data,oulu_con_data)).T
oulu_labels_ad=np.ones((np.shape(oulu_ad_data)[1],1))
oulu_labels_con=np.zeros((np.shape(oulu_con_data)[1],1))
oulu_labels=np.vstack((oulu_labels_ad,oulu_labels_con))
idx = np.random.permutation(len(oulu_labels))
lst_oulu,oulu_labels = lst_oulu[idx], oulu_labels[idx]


mask,model_,model_type,weights=create_mask(data=lst_oulu,labels=oulu_labels,number_of_cv=5,
                                    feature_selection_type='L2_penality',Hyperparameter=1000,mask_threshold=0,model_type='gaussian_process')
print(np.shape(mask))
print(model_)
print(model_type)
print(np.shape(weights))
'''
