%Compute what percentange of training sample classified correctly.
function rate = correct_class(D,A,X,cat_labels,Datom_labels)

%Dictionary, training data matrix dimensions
[~,M] = size(X);
%[~,N] = size(D);

%For later convenience, sift out indices for each class, use to 
%construct delta operators.
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

%In the end, what fraction of training sample correctly
%classified?
classif = zeros(1,M);
thing = zeros(1,C); %Working space
succ = 0;
for i=1:M %Loop over samples
    for j=1:C
        alpha = delta{j}(A(:,i));
        thing(j) = norm(D*alpha-X(:,i),2);
    end
    [~,urp] = min(thing);
    classif(i) = urp;
    if classif(i) == cat_labels(i)
        succ = succ+1;
    end
end

% fprintf("%f classified correctly\n", 100*succ/M);
rate = 100*succ/M;