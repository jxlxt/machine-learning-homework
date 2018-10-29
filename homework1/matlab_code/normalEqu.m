function [theta] = normalEqu(x,y,q,lambda)
% theta = zeros(q, 1);
I = eye(q);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

theta=(((x'*x)+lambda*eye(q)))\(x'*y);


% -------------------------------------------------------------


% ============================================================

end