function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

xVec = theta' * X';
Hyp = sigmoid(xVec);
vecSum = 0;
for i = 1:m
	vecSum = vecSum + (((-1 * y(i)) * log(Hyp(i))) - ((1 - y(i)) * log(1 - Hyp(i))));
end
thSum = sum(theta(2:end) .^ 2);
J = ((1/m) * vecSum) + ((lambda / (2 * m)) * thSum);

[rows, cols] = size(X);
for j = 2:cols
	indSum = 0;
	for i = 1:rows
		indSum = indSum + ((Hyp(i) - y(i)) * X(i, j));
	end
	grad(j) = ((1/m) * indSum) + ((lambda / m) * theta(j));
end
inSum = 0;
for k = 1:rows
	inSum = inSum + ((Hyp(k) - y(k)) * X(k, 1));
end
grad(1) = (1/m) * inSum;



% =============================================================

end
