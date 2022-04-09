[trainingdata, traingnd] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
trainingdata = double(reshape(trainingdata, size(trainingdata,1)*size(trainingdata,2), []).');
X=trainingdata';
traingnd = double(traingnd);
label1=traingnd';
M = size(X,2);
% %Normalize training data to unit length
% for k=1:M
%     X(:,k) = X(:,k)/norm(X(:,k),2);
% end
% save('trainingdata.mat','X','label1');

X2=X(:,1:M/10);
label11=label1(:,1:M/10);
save('small-trainingdata.mat','X2','label11');


[testdata, testgnd] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
testdata = double(reshape(testdata, size(testdata,1)*size(testdata,2), []).');
T=testdata';
testgnd = double(testgnd);
label2=testgnd';
save('testingdata.mat','T','label2')

d = 784;
N = 100;

%Dictionary, training data matrix dimensions
[d,M] = size(X);

D = zeros(d,N);
Dlabel = zeros(1,N);

% 0
ccc = 1;
for j=0:9
    for k=1:M
        if(label1(k)==j)
            D(:,ccc) = X(:,k);
            Dlabel(ccc)=j;
            ccc=ccc+1;
        end
        if(ccc>(N-1)*(j+1)/10+1) break;
        end
    end
end


% for i=0:N-1
%     Dlabel(i+1) = floor(i/2);
% end

% %0-4
% D(:,1) = X(:,2);
% D(:,2) = X(:,22);
% D(:,3) = X(:,4);
% D(:,4) = X(:,7);
% D(:,5) = X(:,6);
% D(:,6) = X(:,17);
% D(:,7) = X(:,8);
% D(:,8) = X(:,11);
% D(:,9) = X(:,3);
% D(:,10) = X(:,10);
% %5-9
% D(:,11) = X(:,12);
% D(:,12) = X(:,36);
% D(:,13) = X(:,14);
% D(:,14) = X(:,19);
% D(:,15) = X(:,16);
% D(:,16) = X(:,30);
% D(:,17) = X(:,18);
% D(:,18) = X(:,32);
% D(:,19) = X(:,20);
% D(:,20) = X(:,23);



% %Normalize atoms to unit length
% for k=1:N
%     D(:,k) = D(:,k)/norm(D(:,k),2);
% end

save('mid-dictionary.mat','D','Dlabel')