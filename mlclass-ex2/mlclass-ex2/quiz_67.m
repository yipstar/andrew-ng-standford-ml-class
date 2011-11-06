x = [5; -5; -5;];

Theta1 = [1 0 1; 1 0 1; 1 1 0];

a2 = zeros (3, 1);

for i = 1:3

  for j = 1:3
    a2(i) = a2(i) + x(j) * Theta1(i, j);
  end

  a2(i) = sigmoid (a2(i));
end

%a2

% choice 2, wrong.
%z = sigmoid(x);
%a2 = Theta1 * z;

% choice 3, wrong.
%z = sigmoid(x); a2 = Theta1 * z;
%a2

% choice 4, correct.
a2 = sigmoid (Theta1 * x);

% choice 5, wrong
%z = sigmoid(x); 
%a2 = sigmoid (Theta1 * z);
%a2

Theta1 = [1 -1.5 3.7; 1 5.1 2.3];
Theta2 = [1 0.6 -0.8];

a2 = sigmoid (Theta1 * x)

new_Theta1 = [1 5.1 2.3; 1 -1.5 3.7];
new_Theta2 = [1 -0.8 0.6];

a2 = sigmoid (new_Theta1 * x)

