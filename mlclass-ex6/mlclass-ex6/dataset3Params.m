function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

num_steps = 8;
steps = [.01; .03; .1; .3; 1; 3; 10; 30];

best_prediction_error = 0;

for c_i=1:num_steps 

  try_C = steps(c_i);

  for s_i=1:num_steps 

    try_sigma = steps(s_i);

    % train the model with current C and sigma parameters.
    model = svmTrain(X, y, try_C, @(x1, x2) gaussianKernel(x1, x2, try_sigma));

    % get the predictions for this run.
    predictions = svmPredict(model, Xval);

    % compute prediction error
    prediction_error = mean(double(predictions ~= yval));    

    if best_prediction_error == 0
      best_prediction_error = prediction_error;
    end

    if prediction_error < best_prediction_error

      best_prediction_error = prediction_error;
      
      % store current values of C and sigma if we're the lowest error so far.
      C = try_C;
      sigma = try_sigma;
    end

  end
end


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
