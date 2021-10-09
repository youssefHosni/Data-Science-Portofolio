function [scaledANDcleanedData] = scalingANDoutliers(data,scaling,outliers)
%Scales the activity data person-wise and postition-wise
%   Detailed explanation goes here

%positions = {'leftPocket','rightPocket','belt','wrist','upperArm'};

%scaling

    
    if scaling == 1 %standardization
       
        minVal = min(data,[],2);
        maxVal = max(data,[],2);

        data = (data - minVal)./maxVal;
        
         
    else
    ;
    end
    
    if outliers == 1
    
        if max(data.(positions{f})) > 2
            
           data.(positions{f})(data.(positions{f}) > 2) = NaN; 
            
        end
    
    else
    ;
    end


scaledANDcleanedData = data;

end


