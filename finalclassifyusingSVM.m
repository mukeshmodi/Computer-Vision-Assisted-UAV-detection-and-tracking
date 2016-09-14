clear all
clc
load('bird_feature_vector.mat');
load('uav_feature_vector.mat');
training_data =[feature_vector(:,1:3500),uav_feature_vector(:,1:3500)];
test_data=[feature_vector(:,3501:end),uav_feature_vector(:,3501:end)];
training_label=[zeros(3500,1);ones(3500,1)];
test_label=[zeros(497,1);ones(500,1)];
test_data=test_data';
sv=svmtrain(training_data,training_label','kernel_function','rbf');
%svm_rbf=svmtrain(training_data,training_label,'kernel_function','rbf');
% svm_quad=svmtrain(total_feature,label_matrix,'kernel_function','quad');
out=[];

   for i=1:497
    ou=svmclassify(sv,test_data(i,:));
    out=[out,ou];
   end
    count=1;
for k=1:length(out)
    if(out(k)==test_label(k))
        count=count+1;
    end
end
acc=count/length(test_label);