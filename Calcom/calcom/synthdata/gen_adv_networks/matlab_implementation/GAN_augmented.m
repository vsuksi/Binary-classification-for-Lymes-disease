function Data = GAN_augmented(Train,epoch)
generator = nnsetup([30, 52, 224]);
discriminator = nnsetup([224, 52, 1]);

batch_size = 30;
item_num = size(Train,1); % suppose training set is m by n, m = numbers of items
batch_num = item_num / batch_size;
learning_rate = 0.01;

Data = [];
for e=1:epoch
    kk = randperm(item_num);
    for t=1:batch_num
        
        data_real = Train(kk((t - 1) * batch_size + 1:t * batch_size), :, :);
        noise = normrnd(0, 0.1, batch_size, 30);
       % generator discriminator
        generator = nnforward(generator, noise);
        data_fake = generator.layers{generator.n}.a;
        discriminator = nnforward(discriminator, data_fake);
        label_fake = discriminator.layers{discriminator.n}.a;
        discriminator = nnbp_discriminator(discriminator, label_fake, ones(batch_size, 1));
        generator = nnbp_generator(generator, discriminator);
        generator = nngradient(generator, learning_rate);
        %---------------------------------------------------
        generator = nnforward(generator, noise);
        data_fake = generator.layers{generator.n}.a;
        discriminator = nnforward(discriminator, data_fake);
        label_fake = discriminator.layers{discriminator.n}.a;
        discriminator = nnbp_discriminator(discriminator, label_fake, zeros(batch_size, 1));
        dw1_t = discriminator.layers{1}.dw;
        db1_t = discriminator.layers{1}.db;
        dw2_t = discriminator.layers{2}.dw;
        db2_t = discriminator.layers{2}.db;
        discriminator = nnforward(discriminator, data_real);
        label_real = discriminator.layers{discriminator.n}.a;
        discriminator = nnbp_discriminator(discriminator, label_real, ones(batch_size, 1));
        discriminator = nngradient(discriminator, learning_rate);
        discriminator = nngradient2(discriminator, learning_rate, dw1_t, db1_t, dw2_t, db2_t);
        % ----------------loss
        if t == batch_num
            c_loss = sigmoid_cross_entropy(label_fake, ones(batch_size, 1));
            d_loss = sigmoid_cross_entropy(label_fake, zeros(batch_size, 1)) + sigmoid_cross_entropy(label_real, ones(batch_size, 1));
            fprintf('c_loss:"%f",d_loss:"%f"\n',c_loss, d_loss);
            Data = [data;data_fake];
        end
        
    end
end




end

