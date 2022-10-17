import pdb

def gan(data,**kwargs):
    '''
    Function to generate synthetic data using Generative Adversarial Networks.

    Inputs:
        data - m-by-n numpy array of data to be used for training
    Outputs:
        fakedata - additional fake data generated through GAN
    Optional inputs:
        epoch           - integer. Number of epochs (outer iterations) for training the network. (default: 100)
        batch_size      - integer. Number of data points to feed in at a time. (default: 30)
        learning_rate   - float. Step size for steepest descent. (default: 0.01)
        noise_dim       - integer. Number of dimensions of noise to input to the generator (default: 30)

        automatic_layer_structure   - boolean. If True, options below are ignoed. (default: False)
        gen_arch        - list of integers indicating network architecture of generator
        dis_arch        - list of integers indicating network architecture of discriminator

        verbosity       - integer. Degree of print statements (default: 0)
    '''
    import numpy as np
    from nnsetup import NN
    from nnbp_discriminator import nnbp_discriminator
    from nnbp_generator import nnbp_generator
    from sigmoid_cross_entropy import sigmoid_cross_entropy

    # TEMPORARY-------------

    global fig
    global idx
    global ax
    global cols
    global ad
    global eq

    pyplot.ion()

    gen_arch = kwargs.get('gen_arch',[])
    dis_arch = kwargs.get('dis_arch',[])
    als = kwargs.get('automatic_layer_structure', False)

    epoch = kwargs.get('epoch', 100)
    batch_size = kwargs.get('batch_size', 30)
    learning_rate = kwargs.get('learning_rate', 0.01)
    noise_dim = kwargs.get('noise_dim', 30)

    verbosity = kwargs.get('verbosity', 0)

    # generator = nnsetup([30,52,224])      # Why is this structured like this?
    # discriminator = nnsetup([224,52,1])   # Presumably fake data in R^224 mapped to 1.

    m,n = np.shape(data)
    batch_num = m//batch_size

    if len(gen_arch)==0 or als:
        gen_arch = [noise_dim,max(n//2,1),n]
    if len(dis_arch)==0 or als:
        dis_arch = [n,max(n//2,1),1]
    #

    generator = NN(gen_arch)
    discriminator = NN(dis_arch)

    fakedata = np.empty((0,n))
    plotcolls = []
    ncolls = 16
    for e in range(epoch):
        # pdb.set_trace()

        kk = np.random.permutation(m)   # Shuffle the data.
        for t in range(batch_num):
            data_real = data[kk[t*batch_size:(t+1)*batch_size]]
            noise = 0.1*np.random.randn(batch_size, noise_dim)

            # ----------------------------------------
            # Generator discriminator phase

            # Propagate the noise through the network and pull out
            # the fake data samples.
            generator.nnforward(noise)
            data_fake = generator.layers[-1].a

            # Propagate the fake data through the discriminator
            # and get the judgment for whether the data is fake or not.
            discriminator.nnforward(data_fake)
            label_fake = discriminator.layers[-1].a

            # Back-propagation through the discriminator network.
            discriminator = nnbp_discriminator(discriminator, label_fake, np.ones( (len(label_fake),1)) )

            # Back-propagation through the generator network.
            generator = nnbp_generator(generator, discriminator)

            # Steepest descent iteration
            generator.nngradient(learning_rate)
            #------------------------------------------

            # Generate fake data again. (?)
            generator.nnforward(noise)
            data_fake = generator.layers[-1].a

            # Propagate the fake data through the discriminator.
            discriminator.nnforward(data_fake)
            label_fake = discriminator.layers[-1].a

            # Back-propagation through the discriminator network. What distinguishes this from the previous run,
            # other than ones or zeros?
            discriminator = nnbp_discriminator(discriminator, label_fake, np.zeros( (batch_size,1) ) )

            # Get step sizes.
            dw1_t = discriminator.layers[0].dw
            db1_t = discriminator.layers[0].db
            dw2_t = discriminator.layers[1].dw
            db2_t = discriminator.layers[1].db

            # Push the real data through the discriminator.
            discriminator.nnforward(data_real)
            label_real = discriminator.layers[-1].a

            # Update the discriminator based on the real examples.
            discriminator = nnbp_discriminator( discriminator, label_real, np.ones( (batch_size,1) ) )
            discriminator.nngradient(learning_rate)
            discriminator.nngradient2(learning_rate, dw1_t, db1_t, dw2_t, db2_t) # ??????????????

            if t==batch_num-1:
                c_loss = sigmoid_cross_entropy(label_fake, np.ones((batch_size,1)))
                d_loss = sigmoid_cross_entropy(label_fake, np.zeros((batch_size,1))) + sigmoid_cross_entropy(label_real, np.ones((batch_size,1)))
                if verbosity>0:
                    print('c_loss: %.3e, d_loss: %.3e'%(c_loss,d_loss))
                #
                fakedata = np.vstack( (fakedata, data_fake) )

                ax.cla()
                ax.scatter(data[:,0], data[:,1], s=40, marker='o', c='k')
                if len(plotcolls)<ncolls:
                    plotcolls.append([data_fake,cols[idx%16]])
                else:
                    plotcolls.pop(0)
                    plotcolls.append([data_fake,cols[idx%16]])
                #
                for coll in plotcolls:
                    ax.scatter(coll[0][ad[0],:], coll[0][ad[1],:], s=40, marker='x', c=coll[1])
                idx += 1
                # print(cols[idx%16])
                # pyplot.show(block=False)
                pyplot.draw()
                pyplot.pause(0.1)  # Force drawing
            #
        #

    #

    return fakedata
#

def gen_fake_data(m,n,eps=0.1):
    '''
    Makes fake labeled data which is correlated and highly separable in
    two dimensions, and white noise in the remaining dimensions.

    Points are randomly selected from the two classes, but should be roughly equal.

    Inputs:
        m       - integer, number of data points
        n       - integer, dimensionality of the problem.
    Optional inputs:
        eps     - float, relative magnitude of noise relative to the magnitude
                    of the weights in the "separable" dimensions (default: 0.1)
    Outputs:
        data        - numpy array of shape (m,n)
        labels      - numpy array of shape m.
        active_dims - numpy array of shape 2, indicating the separable dimensions.
    '''
    active_dims = np.random.choice(np.arange(n), 2, replace=False)

    data = eps*np.random.randn(m,n)
    labels = np.random.choice([0,1], m)

    mus = np.array([[-1.,-1.], [1.,1.]])    # Means of the two classes' distributions
    stds = np.array([0.7,0.7])                # For now, simple covariance in each class.

    for i,l in enumerate(labels):
        data[i,active_dims] = mus[l] + stds[l]*np.random.randn(2)
    #

    return data,labels,active_dims
#

# ----------- TEST IT OUT ----------------------
if __name__=="__main__":
    import numpy as np
    from matplotlib import pyplot
    m,n = 1000,10

    global fig
    global idx
    global ax
    global cols
    global ad
    global eq

    data,labels,ad = gen_fake_data(m,n)
    eq = {l: np.where(labels==l)[0] for l in np.unique(labels)}

    fig,ax = pyplot.subplots(1,1)
    # Real data
    ax.scatter(data[eq[0],0], data[eq[0],1], s=40, marker='o', c='k')
    pyplot.show(block=False)
    pyplot.ion()

    # Fake data
    idx=0
    bs = 30
    cols = pyplot.cm.rainbow(np.linspace(0,1,16))



    fakedata = gan(data[eq[0]], epoch=10000, batch_size=10)

    # while True:
    #     # ax.scatter(fakedata[idx*bs:(idx+1)*bs,ad[0]], fakedata[idx*bs:(idx+1)*bs,ad[1]], s=40, marker='x', c=cols[idx%16]*np.exp(-0.002*idx))
    #     ax.scatter(fakedata[idx*bs:(idx+1)*bs,ad[0]], fakedata[idx*bs:(idx+1)*bs,ad[1]], s=40, marker='x', c=cols[idx%16])
    #     idx += 1
    #     if (idx+1)*bs >= fakedata.shape[0]:
    #         break
    # #

    pyplot.show(block=False)
#
