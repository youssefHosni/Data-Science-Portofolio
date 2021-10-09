function [acc,class_accuarcy] = classification(train_features,train_labels,test_features,test_labels,classifier_name)
%Apply the croos validation on the data and use Knn classifier 
%   Detailed explanation goes here


    if strcmp(classifier_name,'KNN')
        clear classifer_model;
        classifer_model=fitcknn(train_features,train_labels,'NumNeighbors',5);
        predicted_labels=predict(classifer_model,test_features);
        performance_evaluaion=classperf(test_labels);
        classperf(performance_evaluaion,predicted_labels);
        acc=length(find(predicted_labels==test_labels))/length(predicted_labels);
        class_accuarcy = classes_accuarcy(test_labels,predicted_labels);
   
    elseif strcmp(classifier_name,'LDA')
        clear classifer_model;
        classifer_model=fitcdiscr(train_features,train_labels,'DiscrimType','linear');
        predicted_labels=predict(classifer_model,test_features);
        acc=length(find(predicted_labels==test_labels))/length(predicted_labels);
        class_accuarcy = classes_accuarcy(test_labels,predicted_labels);

    
    elseif strcmp(classifier_name,'QDA')
        classifer_model=fitcdiscr(train_features,train_labels,'DiscrimType','quadratic');
        predicted_labels=predict(classifer_model,test_features);    
        acc=length(find(predicted_labels==test_labels))/length(predicted_labels);
        class_accuarcy = classes_accuarcy(test_labels,predicted_labels);

end

