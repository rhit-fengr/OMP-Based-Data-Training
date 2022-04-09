%Code to perform classification on handwritten digits data.
clear;
clc;

tic;

%Maximum number of iterations to improve dictionary
maxits = 10;

%Regularization parameter
%reg = 0.0;

%Sparsity limit for OMP
sparsity = 5;

%OMP parameters
opts.omp = 1; %Use OMP
opts.maxits = sparsity; %Sparsity limit
opts.verbose = 0;

%Load in training data
load('trainingdata.mat');

%Normalize all columns, any other preprocessing.
[n1,n2] = size(X);
for k=1:n2
    X(:,k) = X(:,k)/norm(X(:,k),2);
end

%Load in testing data
load('testingdata.mat');

%Normalization for testing data
[t1,t2] = size(T);
for k=1:t2
    T(:,k) = T(:,k)/norm(T(:,k),2);
end

%Partition the training data into a dictionary D
%and proper training data X.
D1 = 100; N1 = 2000;
D = X(:,1:D1); %About 10 percent of each digit appears.
%It will be convenient to use the label "1" for the digit "0"
%and more generally label "i" for digit "i-1".
Datom_labels = 1+label1(1:D1);


%Relegate the remaining training data to be actual training data.
X = X(:,D1+1:N1);
cat_labels = 1+label1(D1+1:N1);

Tcat_labels = 1+label2(:,:);

clear label1 label2;

%Dictionary, training data matrix dimensions
[d,M] = size(X);
[~,N] = size(D);
[~,MT] = size(T);

%For later convenience, sift out indices for each class, use to 
%construct delta operators. Note that F{i} is for the digit i-1.
C = max(cat_labels); %Number of classes
F = cell(1,C);

for i=1:C
    F{i} = (Datom_labels.*(Datom_labels==i)/i)'; %These should be column vectors
end

%This function will restrict support of a vector to appropriate
%class label indices.
delta = cell(1,C);
for i=1:C
    delta{i} = @(v) v.*F{i};
end

%Iterate to improve dictionary. The matrix A is as described
%previously in a LaTeX document, the solutions to D*A = X.
A = zeros(N,M);
AT = zeros(N,MT);

for jj=1:maxits
  %Run OMP on each sample. Install results into matrix "A".

  for i=1:M
    %Call OMP to solve D*alpha = x
    [alpha, rhist] = omp(D, X(:,i), opts);
    
    %Restrict support of alpha
    A(:,i) = delta{cat_labels(i)}(alpha);
  end

  for i=1:MT
    %Call OMP to solve D*alpha = x
    [alpha, rhist] = omp(D, T(:,i), opts);
    
    %Restrict support of alpha
    AT(:,i) = delta{Tcat_labels(i)}(alpha);
  end
  
  %Routine to compute percentage correctly classified.
  fprintf("[Pre-iteration %d] %f percent of training data is classified correctly\n", ...
      jj,correct_class(D,A,X,cat_labels,Datom_labels));
  fprintf("                  %f percent of testing data is classified correctly\n", ...
      correct_class(D,AT,T,Tcat_labels,Datom_labels));
  

  

  %How good is this dictionary?
  ED = norm(D*A-X,'fro')/norm(X,'fro');
  fprintf("                  Dictionary error metric is %f\n", ED);

  %Now let's optimize the dictionary
  Dopt = optD(A, X);

  ED2 = norm(Dopt*A-X,'fro')/norm(X,'fro');
  fprintf("                 Improved dictionary error metric is %f\n", ED2);
  
  %Update dictionary
  D = Dopt;

  %Renormalize columns
  for k=1:N
      D(:,k) = D(:,k)/norm(D(:,k),2);
  end

  
  %Compute percent correctly classified, print
  fprintf("[Post-iteration %d] %f percent of training data is classified correctly\n", ...
      jj,correct_class(D,A,X,cat_labels,Datom_labels));
  fprintf("                  %f percent of testing data is classified correctly\n", ...
      correct_class(D,AT,T,Tcat_labels,Datom_labels));

end

toc;