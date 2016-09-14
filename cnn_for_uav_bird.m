load uav_data;
tr_x=[training_uav,train_birds];
te_x=[test_uav,test_birds];


tra_x=reshape(tr_x,[64,64,7000]);
test_x=reshape(te_x,[64,64,1000]);

tra_y = zeros(7000,2);
test_y = zeros(1000,2);

tra_y(1:3500,1)=1;
tra_y(3501:end,2)=1;

train_y=zeros(7000,2);

test_y(1:500,1)=1;
test_y(501:end,2)=1;

v=randperm(7000);
train_x=zeros(64,64,7000);
for k=1:7000
    ind=v(k);
    train_x(:,:,k)=tra_x(:,:,ind);
    train_y(k,:)=tra_y(ind,:);
end 






%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 100;
opts.numepochs = 100;

cnn = cnnsetup(cnn, train_x, train_y');
cnn = cnntrain(cnn, train_x, train_y', opts);

[er, bad] = cnntest(cnn, test_x, test_y');

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');

