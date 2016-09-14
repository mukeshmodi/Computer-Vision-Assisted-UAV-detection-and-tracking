clear all
load('bird_feature_vector.mat');
load('uav_feature_vector.mat');
[r1,c1]=size(feature_vector);
[r2,c2]=size(uav_feature_vector);
training_data =[feature_vector(:,1:3500),uav_feature_vector(:,1:3500)];
test_data=[feature_vector(:,3501:end),uav_feature_vector(:,3501:end)];
training_label=[zeros(3500,1);ones(3500,1)];
test_label=[zeros(497,1);ones(500,1)];
rf_smile=fitensemble(training_data',training_label,'Bag',3500,'tree','type','classification');
out=rf_smile.predict(test_data');
count=0;
for k=1:497
    if(out(k)==test_label(k))
        count=count+1;
    end
end
acc=count/length(test_label);