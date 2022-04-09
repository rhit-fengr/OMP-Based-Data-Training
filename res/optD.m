function optD = optD(A, X)

% delta = 5;

%v1
optD = X/A;

% %v2
% Q = A*A';
% 
% R = X*A';
% 
% optD = R/Q;

%v3
% Q = A*A';
% Q = Q+delta*eye(size(Q));
% 
% R = X*A';
% 
% %May want pseudoinverse?
% optD = R/Q;
