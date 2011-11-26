function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% transform y from a 5000 dim vector into a 5000 x 10 matrix


yv = 1:num_labels;
yv = repmat(yv,m,1);
for i = 1 : m
  yv(i,:) = yv(i,:) == y(i);
end

% 5000 x 10
size(yv);

% feed forward
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
% TODO: could have used this as well?:
% h1 = predict(Theta1, Theta2, X);

% 5000 x 10
a3 = h2;
size(a3);

% 1 x 10
a3_of_i = a3(1, :);
size(a3_of_i);

% 10 x 1
yki1 = yv(1, :)'; 
size(yki1);

% 10 x 1
hxik1 = h2(1, :)';
size(hxik1);

for i=1:m
    yki = yv(i, :)';
    hxik = h2(i, :)';

    J = J + ( (-yki' * log(hxik)) - ((1 - yki') * log(1 - hxik)) ) ;
end

J = 1/m .* J;

% Theta1 - 25 x 401
% Theta2 - 10 x 26 

% 400
input_layer_size;

% 25
hidden_layer_size;

% regularize
reg = 0;

left = 0;
for j=1:hidden_layer_size
  for k=1:input_layer_size + 1
    if k == 1
      left = left;
    else
      left = left + Theta1(j,k)^2;
    end
  end
end
left = lambda/(2*m) * left;

right = 0;
for j=1:num_labels
  for k=1:hidden_layer_size + 1
    if k == 1
      right = right;
    else
      right = right + Theta2(j,k)^2;
    end
  end
end
right = lambda/(2*m) * right;

reg = left + right;
J = J + reg;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%labels = 1:num_labels;

% y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1 (page 5 of ex4.pdf)
yk = zeros(num_labels, m); 
for i=1:m,
  yk(y(i),i)=1;
end

for t=1:m

  %fprintf("i:%i", t);

  % 401 x 1
  a1 = [1; X(t, :)'];

  % 25 x 401 * 401 x 1
  z2 = Theta1 * a1;

  % 25 x 1
  a2 = sigmoid(z2);
  
  % add bias, 26 x 1
  a2 = [1; a2];

  % 10 x 26 *  26 x 1 = 10 x 1
  z3 = Theta2 * a2;
  
  % 10 x 1
  a3 = sigmoid(z3);

  % now comes back prop...

  % TODO: figure out what was wrong with this way of computing yk
  % 10 x 1 
  %yk = (labels == y(i))';

  z2=[1; z2]; % bias

  % 10 x 1
  delta3 = a3 - yk(:, t);

  % 26 x 10 * 10 x 1 = 26 x 1
  delta2 = Theta2' * delta3 .* sigmoidGradient(z2);

  % remove delta2_zero, makes 25 x 1;  
  delta2 = delta2(2:end);

  Theta2_grad = Theta2_grad + delta3 * a2';
  Theta1_grad = Theta1_grad + delta2 * a1';
end

% obtain unregularized gradients
Theta1_grad = 1/m .* Theta1_grad;
Theta2_grad = 1/m .* Theta2_grad;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% zero bias units.
mask1 = ones(size(Theta1));
mask1(:, 1) = 0;
mask2 = ones(size(Theta2));
mask2(:, 1) = 0;

Theta1_grad = Theta1_grad + (lambda/m) * (Theta1 .* mask1);
Theta2_grad = Theta2_grad + (lambda/m) * (Theta2 .* mask2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
