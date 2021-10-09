function [class_accuarcy] = classes_accuarcy(test_labels,predicted_labels)
% giving the predicted labels and the groun truth labels, this function
% returns the class(activity) that is best classified with the classifier

misclassified_index=find(test_labels~=predicted_labels);

misclassified_labels=test_labels(misclassified_index);

unique_labels=unique(test_labels);
class_accuarcy=[]; 
for i =1: length(unique_labels)
counter=length(find(misclassified_labels==unique_labels(i)));
class_accuarcy=[class_accuarcy counter];
end 
class_accuarcy=1-(class_accuarcy/length(test_labels));


end

