function output_index = feature_filter(input_feature_index)
%Find the most notable 3 features to be selected out and print
if size(input_feature_index,1)>=3
    output_index = input_feature_index(1:3,:);
else
    output_index = input_feature_index;
end

end

