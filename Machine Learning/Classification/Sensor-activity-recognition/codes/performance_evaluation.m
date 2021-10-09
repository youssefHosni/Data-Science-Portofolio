function [cp] = performance_evaluation(classifier_model,test_data,test_labels)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
         predicted_labels=predict(classifer_model,test_data);
        cp=classperf(test_labels);
        classperf(cp,predicted_labels)
        


end

