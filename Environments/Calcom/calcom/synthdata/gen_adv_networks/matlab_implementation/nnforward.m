function nn = nnforward(nn, x)
    nn.layers{1}.a = x;
    for i = 2 : nn.n
        a = nn.layers{i-1}.a;
        w = nn.layers{i-1}.w;
        b = nn.layers{i-1}.b;
        % nn.layers{i}.a = a*w + b;
        nn.layers{i}.a = a*w + repmat(b, size(a, 1), 1);
        nn.layers{i}.a = sigmoid(nn.layers{i}.a);
    end
end


