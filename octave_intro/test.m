

v = zeros(10,1)
for i=1:10,
  v(i) = 2^i;
  disp(i);
end
disp(v);

i = 1;
while i <= 5,
  v(i) = 100;
  i = i+1;
end;
v

squareThisNumber(5)

[a,b] = squareAndCubeThisNumber(5)

X = [1 1; 1 2; 1 3]
y = [1; 2; 3]

theta = [0;1]

j = costFunctionJ(X, y, theta)

theta = [0;0]

j = costFunctionJ(X, y, theta)

prediction = 0.0;

% Unvectorized implementation.
%for j = 1:n+1,
%  prediction = prediction + theta(j) * x(j)
%end;

% Vectorized implementation (higly optimzied linear algerba routines)
prediction = theta' * x;
  