function nn = nngradient2(nn, learning_rate, dw1, db1, dw2, db2)
    nn.layers{1}.w = nn.layers{1}.w - learning_rate * dw1;
    nn.layers{1}.b = nn.layers{1}.b - learning_rate * db1;
    nn.layers{2}.w = nn.layers{2}.w - learning_rate * dw2;
    nn.layers{2}.b = nn.layers{2}.b - learning_rate * db2;
end

