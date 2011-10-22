%% Initialization
clear all; close all; clc

data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

plotData(X, y);

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

% initialize fitting parameters
theta = zeros(2, 1);

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost
computeCost(X, y, theta)

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));