function nn = nnsetup(architecture)
    nn.size   = architecture;
    nn.n = numel(nn.size);
    for i = 2 : nn.n   
        nn.layers{i-1}.w = normrnd(0, 0.1, nn.size(i-1), nn.size(i));
        nn.layers{i-1}.b = zeros(1, nn.size(i));
    end
end

