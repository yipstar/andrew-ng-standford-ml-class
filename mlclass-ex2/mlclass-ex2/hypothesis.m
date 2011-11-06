function h = hypothesis(theta, X)
    h = sigmoid(theta' .* X);
end
