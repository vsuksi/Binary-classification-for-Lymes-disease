function nn = nnbp_discriminator(nn, y_h, y)
    
    n = nn.n;
    
    nn.layers{n}.d = -(y .*(1 ./ y_h) + (y-1).*(1 ./ (1-y_h))) .* (y_h .* (1-y_h));
    for i = n-1:-1:1
        d = nn.layers{i+1}.d;
        w = nn.layers{i}.w;
        if i ~= 1
            a = nn.layers{i}.a;
            
            nn.layers{i}.d = d*w' .* (a .* (1-a));
        else
           
            nn.layers{i}.d = d*w';
        end
    end
    
    for i = 1:n-1
        d = nn.layers{i+1}.d;
        a = nn.layers{i}.a;
        
        nn.layers{i}.dw = a'*d / size(d, 1);
        nn.layers{i}.db = mean(d, 1);
    end
end

