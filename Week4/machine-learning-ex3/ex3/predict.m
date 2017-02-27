function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias ones to the X data matrix
X = [ones(m, 1) X];
%Theta1 is 25 x 401; X is m x 401
%hidden is m x 25
hidden = sigmoid(X*Theta1');
% Add bias ones to the hidden data matrix
%hidden is now m x 26
hidden = [ones(m, 1) hidden];
%Theta2 is 10 x 26
%output is m x 10
output = sigmoid(hidden*Theta2');
%p is m x 1 matrix of predicted Digits
[W, p] = max(output, [], 2);




% =========================================================================


end
