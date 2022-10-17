function nn = nngradient(nn, learning_rate)
    n = nn.n;
    for i = 1:n-1
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;
        nn.layers{i}.w = nn.layers{i}.w - learning_rate * dw;
        nn.layers{i}.b = nn.layers{i}.b - learning_rate * db;
    end
end

