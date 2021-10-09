%%  create new feature map and use it 
clear all
clc;
activity_data=load('dataActivity');
classification_accuarcy=[];
k_folds=10; % number of folds used for cross validation
window_size=8; % widnow size for creating the feature map 
create_new_feature_map=1; % if ==1 then a new feature map will be created , else  a saved one will be used 
saved_feature_map_file_name='feature_map_3s.mat'; % the name of the feature map file 
scaling=0; % scaling should be ==1 if you would like to scale the data and 0 if not
outliers=0; % outliers should be ==1 if you would like to remove the outliers and zero if ypu donot like.
%[activity_data] = scalingANDoutliers(activity_data,scaling,outliers);

% Check whether a new feature map will be created or a save one should be
% used
if create_new_feature_map==1
feature_map=create_feature_map(activity_data,window_size);
participant_names=fieldnames(feature_map);
else   
feature_map=load(saved_feature_map_file_name);
feature_map=feature_map.feature_map;
participant_names=fieldnames(feature_map); 
end


classifier_name='KNN'; % classifier name, to change the classifier used check the names of the avaliable classifier from classification function
train_labels=[];
train_features=[];
cross_validation_each_posiiton_acc=[];
max_class_acc_each_position=[];
max_class_acc_value=[];


time_starting_feature_index=1;
frequency_starting_feature_index=7;

time_ending_feature_index=6;
frequency_ending_feature_index=11;

time_features=0;
classification_accuarcy=[];
class_acc_all_folds=[];
for j=1:k_folds
    train_features=[];
    test_features=[];
    train_labels=[];
    test_labels=[];
 for i=1:length(participant_names)
   
    if i==length(participant_names)-(j-1)
        participant=getfield(feature_map,cell2mat(participant_names(i)));
        test_labels= participant(:,1);
        test_features=participant(:,2:end);
    else
        participant=getfield(feature_map,cell2mat(participant_names(i)));
        train_labels=vertcat(train_labels,participant(:,1));
        train_features=vertcat(train_features,participant(:,2:end));
    end 
end
%%%% selecting certain feature for example certain  positions or certain
%%%% axis should be done here before giving the data to the classifier.
position_feature_index_all=[];
for sensor=0:8
for position=0:4
if time_features==1
starting_index=time_starting_feature_index;
ending_index=  time_ending_feature_index; 
elseif time_features==0
starting_index=frequency_starting_feature_index;
ending_index=  frequency_ending_feature_index; 
else
 starting_index=1;
 ending_index=11;
end
position_feature_index=linspace(starting_index,ending_index,ending_index-starting_index+1)+11*sensor+position*99;
position_feature_index_all=[position_feature_index_all position_feature_index];
 end
 end

train_features_certain_positions=train_features(:,position_feature_index_all);
test_features_certain_positions=test_features(:,position_feature_index_all);
[train_features_certain_positions] = scalingANDoutliers(train_features_certain_positions,scaling,outliers);
[test_features_certain_positions] = scalingANDoutliers(test_features_certain_positions,scaling,outliers);
[acc,class_acc] =classification(train_features_certain_positions,train_labels,test_features_certain_positions,test_labels,classifier_name);
classification_accuarcy=[classification_accuarcy acc];
class_acc_all_folds=[class_acc_all_folds ; class_acc];
end
cross_validation_acc=mean(classification_accuarcy);
class_acc_one_position=mean(class_acc_all_folds,1);
[max_class_acc,max_class_label]=max(class_acc_one_position);






