function g_net = nnbp_generator(g_net, d_net)
    n = g_net.n;
    g_o = g_net.layers{n}.a;
    
    g_net.layers{n}.d = d_net.layers{1}.d .* (g_o .* (1-g_o));
    for i = n-1:-1:1
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i}.w;
        if i ~= 1
            a = g_net.layers{i}.a;
            g_net.layers{i}.d = d*w' .* (a .* (1-a));
        else
            g_net.layers{i}.d = d*w';
        end
    end
    for i = 1:n-1
        d = g_net.layers{i+1}.d;
        a = g_net.layers{i}.a;
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end


