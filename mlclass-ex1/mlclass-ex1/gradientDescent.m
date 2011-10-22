function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

  predictions = X * theta;
  errors = (predictions - y);

  summation = 0;
  for i = 1:m
    summation = summation + ( (hypothesis(theta, X(i, 2)) - y(i)) .* X(i, 2) );
  end

  temp_theta_zero = theta(1) - (alpha .* (1/m) .* sum(errors));
  temp_theta_one = theta(2) - (alpha .* (1/m) .* summation);

  theta(1) = temp_theta_zero;
  theta(2) = temp_theta_one;
  
  % Save the cost J in every iteration    
  J_history(iter) = computeCost(X, y, theta);

end
