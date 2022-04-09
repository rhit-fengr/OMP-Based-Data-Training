function [xsol, rhist] = omp(A, b, opts)
%Usage: [xsol, rhist] = omp(A, b, opts)
%Given matrix A and vector b, run omp (or just mp) to solve A*x = b.
%Returns solution in "xsol", residual history in "rhist".

%Function to read options
function out = setOpts( field, default )
    if ~isfield( opts, field )
        opts.(field)    = default;
    end
    out = opts.(field);
end

%Scale the columns of A; compensate later.
[m,n] = size(A); %A is m by n
A2 = zeros(size(A));
s = zeros(n,1);
for i=1:n
    s(i) = norm(A(:,i),2);
    A2(:,i) = A(:,i)/s(i);
end

%Set options
do_omp = setOpts('omp', 1); %Default to OMP
verbose = setOpts('verbose', 0);
maxits = setOpts('maxits', floor(m/2));
coher = setOpts('coher', 0);
tol = setOpts('tol', 1.0e-10);

%If requested, computed and print coherence
if coher
    thing = abs(A2'*A2-eye(n));
    coh = max(max(thing));
    fprintf('Coherence is %f\n', coh);
end

k = 0; %Loop counter
xsol = zeros(n,1); %Prospective solution
S = []; %Support indices of tentative solution

%Loop, build up solution one index a time
while k<=maxits
    if verbose
        fprintf('Estimated Solution '); disp(xsol');
        pause
    end
    resid = b - A2*xsol; %Residual vector
    rhist(k+1) = norm(resid,2); %Record its norm
    if verbose
        fprintf('Residual Vector '); disp(resid');
        pause
    end
    %fprintf('Iteration %d residual %f\n',k,rhist(k+1));
    if rhist(k+1) < tol || k==maxits %Quit if small enough
        break;
    end
    c = A2'*resid;
    if verbose
        fprintf('c = AT*r  '); disp(c');
        pause;
    end
    
    [ci,ii] = max(abs(c));
    if verbose
        fprintf('max ci = %f Adding index %d to support\n', ci, ii);
        pause
    end
    
    %Matching Pursuit
    xsol(ii) = xsol(ii) + c(ii);
    S = union(S,ii);
    
    if verbose
        fprintf('Estimated support S = '); disp(S);
        pause
    end
    
    %Or do OMP if flag set
    if do_omp == 1
        %Implement the least squares solution on suppport set S.
        A3 = A2(:,S);
        xsol3 = A3\b;
        xsol(S) = xsol3;
    end
    
    k = k+1;
end

%Adjust xsol back due to scaling
xsol = xsol./s;

end