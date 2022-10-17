function result = sigmoid_cross_entropy(y_h, y)
    result = -(y.*log(y_h) + (1-y).*log(1-y_h));
    result = mean(result);
end

