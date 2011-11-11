function [J grad] = nnCostFunctionSave(nn_params, ...
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

size(Theta1);
size(Theta2);
size(X);
size(y);

p = predict(Theta1, Theta2, X);
size(p);

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
a = 1:10;

new_y = zeros(m, 10);
for i=1:m
  new_y(i, :) = (a == y(i));
end
size(new_y);

% LR Cost function
%J = 1./m * ( -y' * log( sigmoid(X * theta) ) - ( 1 - y' ) * log ( 1 - sigmoid( X * theta)) );

% NN Cost function
h1 = predict(Theta1, Theta2, X);

new_h1 = zeros(m, 10);
for i=1:m
  new_h1(i, :) = (a == h1(i));
end
size(new_h1);

%left = new_y' * log(new_h1);
%size(left)

%right = (1 - new_y)' * log(1 - new_h1);
%size(right)

%whole = left + right;
%size(whole);

%J = 1./m * sum( (-y' * log(h1) - (1 - y') * log(1 - h1) );
%J = 1./m * sum((-new_y' * log(h1) - (1 - new_y)' * log(1 - h1)));
%size(J)
%J

%J = 1./m * ( -new_y' * log( new_h1 ) - ( 1 - new_y' ) * log ( 1 - new_h1 ) );
%size(J)

% J = 0;

%J = -1./m * ( ( y' * log(h1) + (1 - y') * log (1 - h1)) );


%for i=1:m
%  for k=1:num_labels
i=1;
k = 1;
    
    yk = (a == y(i))'
    hxk = (a == h1(i))'

    %left = yk' * hxk
    left = log(hxk)

    %left = (yk' * log(hxk))
    %size(left)

    %right = (1-yk)' * log(1-hxk)
    %size(right)

    %J = J + (left + right)
    %J 

    %J = yk' * hxk
    %J = yk' * log(hxk)

    %yk' * log(hxk)

    %yk * log(hxk)'

    %J = J + yk' * log(hxk)'

%    J = J + ( (yk' * log(hxk)) + (1 - yk)' * log(1 - hxk) );

%  end

%end

%J = J * -1./m;

%J


%for k=1:num_labels

  %yk = zeros(1, 10);
  %yk = (a == k)';

  %hxk = h1;

  %J = J + (1./m * ( -yk' * log( h1(:, k) )      )      )
  %test = -yk' * log (new_h1(:, k));

  %J = J + ( 1./m * ( -yk' * log( new_h1(:, k) ) - ( 1 - new_y' ) * log ( 1 - new_h1(:, k) ) ) );
%end

J

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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
