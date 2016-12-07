function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1.1;
sigma = 0.13;

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
%{
err_min = 1.0;
for C = 0.1:1:10.1
  for sigma = 0.1:0.01:0.2
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    C
    sigma
    err
    if err < err_min
      c_min = C;
      sigma_min = sigma;
      err_min = err;
    end
  end
end

C = c_min;
sigma = sigma_min;
C
sigma
err_min
%}
% =========================================================================

end

function c = costFunc(var, X, y, Xval, yval)

model= svmTrain(X, y, var(1), @(x1, x2) gaussianKernel(x1, x2, var(2)));
predictions = svmPredict(model, Xval);
c = mean(double(predictions ~= yval));
var
c
end
