function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% 12 x 2
size(X);

% 12 x 1
size(y);

% 2 x 1 vector
size(theta);

% 1
lambda;




% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Remember to always use the size function for this and not assume theta dimensions when zeroing bias.
theta_bias_zeroed = [0 ; theta(2:size(theta), :)];

% 12 x 1
h1 = X * theta;
J = 1/(2 * m) * sum((h1 - y).^2);

% don't regularize theta(1) theta0.
reg = lambda / (2 * m) * sum(theta_bias_zeroed.^2);
J = J + reg;

% grad is a 2 x 1 vector
grad = ((X' * (h1 - y)) + lambda * theta_bias_zeroed) / m;

% =========================================================================

grad = grad(:);

end
