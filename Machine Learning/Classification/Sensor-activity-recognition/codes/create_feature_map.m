function [feature_map] = create_feature_map(activity_data,window_size)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
struct_fields=fieldnames(activity_data);
feature_map=struct;
for i=1:length(struct_fields)
    participant=getfield(activity_data,cell2mat(struct_fields(i)));
    labels=participant.labels;
    new_label_index=find((labels(2:end)-labels(1:end-1))~=0);
    new_label_index=[1 ; new_label_index ; length(labels)];
    participant_field_names=fieldnames(participant);
    labels_col=[];
    feat_map_participant=[];
    feature_map_all_positions=[];
    for j=1:length(participant_field_names)
         feat_map_all_labels_one_position=[];
        if ~strcmp(cell2mat(participant_field_names(j)), 'labels') && ~strcmp(cell2mat(participant_field_names(j)), 'time')
            position_data=getfield(participant,cell2mat(participant_field_names(j)));
            labels_col=[];
            for k=1:length(new_label_index)-1
                feat_map_all_axis_one_label=[];
               for c=1:9
                if k==length(new_label_index)-1
                    strip=position_data(new_label_index(k):new_label_index(k+1),c);
                    current_label=labels(new_label_index(k+1));
                            
                else   
                    strip=position_data(new_label_index(k):new_label_index(k+1),c);
                    current_label=labels(new_label_index(k+1));
                end
                feat_map_axis=features(strip,window_size);
                feat_map_all_axis_one_label=horzcat(feat_map_all_axis_one_label,feat_map_axis);
                
               end 
              labels_col_temp=ones(size(feat_map_all_axis_one_label,1),1)*current_label;
              labels_col=vertcat(labels_col,labels_col_temp);
              feat_map_all_labels_one_position=vertcat(feat_map_all_labels_one_position,feat_map_all_axis_one_label);
            end
            feature_map_all_positions=horzcat(feature_map_all_positions,feat_map_all_labels_one_position);
            
        end
    end
    feat_map_participant=horzcat(labels_col,feature_map_all_positions);
    feature_map.(cell2mat(struct_fields(i)))=feat_map_participant;
end
save([ 'feature_map_' int2str(window_size) 's.mat'],'feature_map')
end

