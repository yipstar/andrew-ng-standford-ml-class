A = [1 1 1; 2 2 2; 3 3 3; 4 4 4; 5 5 5];
B = [1 1 1 1 1; 2 2 2 2 2; 3 3 3 3 3];

C = A * B;

C;

R = [1 1 1 1 1; 0 0 0 0 0; 1 1 1 1 1; 0 0 0 0 0; 1 1 1 1 1];

total = 0;
for i = 1:5
  for j = 1:5
    if (R(i,j) == 1)
      total = total + C(i,j);
    end
  end
end

total;

C = (A * B) .* R; total = sum(C(:));

%total = sum(sum(A(R == 1) * B(R == 1)))

total = sum(sum((A * B) .* R))