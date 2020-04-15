function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

for i=1:num_movies
  for j=1:num_users

    if R(i,j) == 1
      
      step_cost = ( (Theta(j, :) * X(i, :)') - Y(i, j) ).^2;
      J = J + step_cost;
    end
  end
end

J = 1/2 .* J;

% regularization

% TODO: why the hell does this not work?
%J = J + ( lambda/2 .* sum(Theta.^2) ) + ( lambda/2 .* sum(X.^2) );

theta_reg = 0;
for j=1:num_users
  for k=1:num_features
    theta_reg = theta_reg + Theta(j,k)^2;
  end
end

theta_reg = lambda/2 * theta_reg;

x_reg = 0;
for i=1:num_movies
  for k=1:num_features
    x_reg = x_reg + X(i,k)^2;
  end
end

x_reg = lambda/2 * x_reg;

J = J + theta_reg + x_reg;

% gradients

for i=1:num_movies
  for j=1:num_users

    if R(i,j) == 1      

%      for k=1:num_features
%        X_grad(i, k) = X_grad(i, k) + ( ((Theta(j, k) * X(i, k)) - Y(i, j)) * Theta(j, k) );
%        Theta_grad(j, k) = Theta_grad(j, k) + ( ((Theta(j, k) * X(i, k)) - Y(i, j)) * X(i, k) );
%      end

%      X_grad(i, :) = X_grad(i, :) + ( ((Theta(j, :) * X(i, :)') - Y(i, j)) * Theta(j, :) );
%      Theta_grad(j, :) = Theta_grad(j, :) + ( ((Theta(j, :) * X(i, :)') - Y(i, j)) * X(i, :) );

%      X_grad(i, :) = X_grad(i, :) + ( X(i, :) * lambda );
%      Theta_grad(j, :) = Theta_grad(j, :) + ( Theta(j, :) * lambda );
    end


  end
end

X_grad = ((X * Theta' - Y) .* R) * Theta + lambda * X;
Theta_grad = ((X * Theta' - Y) .* R)' * X + lambda * Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
