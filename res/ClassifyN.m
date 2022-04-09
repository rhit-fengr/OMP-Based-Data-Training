%Start of code to perform classification on similated data.
clear;
clc;

%Sparsity limit for OMP
sparsity = 5;

%OMP parameters
opts.omp = 1; %Use OMP
opts.maxits = sparsity; %Sparsity limit
opts.verbose = 0;

%Load in training data and class labels
load('small-trainingdata.mat');X=X2;label1=label11;
clear X2; clear label11;

d = 784;
N = 1000;

%Dictionary, training data matrix dimensions
[d,M] = size(X);

%Load in dictionary data and Dlabels
load('dictionary.mat');
% save('dictionary.mat','D','Dlabel')

C = max(label1)+1; %Number of classes
F = cell(1,C);
for i=1:C
    F{i} = (Dlabel.*(Dlabel==i)/i)'; %These should be column vectors
end

%This function will restrict support of a vector to appropriate
%class label indices.
delta = cell(1,C);
for i=1:C
    delta{i} = @(v) v.*F{i};
end

if 0

count=0;
while(count<10)
%Run OMP on each sample. Install results into matrix "A".
A = zeros(N,M);
for i=1:M
    %Call OMP to solve D*alpha = x
    [alpha, rhist] = omp(D, X(:,i), opts);
    
    %Restrict support of alpha
    A(:,i) = delta{label1(i)+1}(alpha);
end

%How good is this dictionary?
ED = norm(D*A-X,'fro')/norm(X,'fro');
fprintf("[%d iteration] Dictionary error metric for D is %f\n", count, ED);
if(count==0)
ED0 = ED;
end

%Now let's optimize the dictionary
Dopt = optD(A, X);
fprintf("dic diff is %f\n",norm(D-Dopt,'fro'));

ED2 = norm(Dopt*A-X,'fro')/norm(X,'fro');
fprintf("[%d iteration] Dictionary error metric for Dopt is %f\n", count, ED2);
fprintf("[%d iteration] |ED-ED2|/ED val: %f \n", count, abs(ED-ED2)/abs(ED))
if(abs(ED-ED2)/abs(ED)<1e-5) 
    EDN = ED2;
    fprintf("[%d iteration] D doesn't get significantly optimized anymore. Finished.\n", count)
    break;
end
D = Dopt;
count=count+1;
end
fprintf("[%d iteration] Finished.\n", count)
fprintf("The performance got improved by %f percent in %d iterations.",abs(ED0-EDN)*100/abs(ED0), count)
end