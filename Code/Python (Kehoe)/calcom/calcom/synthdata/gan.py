# Adapted from
# https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_pytorch.py
#

# We want to get through the first 'import calcom'
# even if they do not have torch. Errors will be thrown
# later for not having torch installed.
try:
    import torch
except ImportError:
    torch = None

if torch:
    import torch.nn.functional as nn
    import torch.autograd as autograd
    import torch.optim as optim
    from torch.autograd import Variable
#

import numpy as np
from calcom.synthdata import AbstractSynthGenerator

class GANGenerator(AbstractSynthGenerator):
    #__metaclass__ = ABCMeta

    def __init__(self):

        if not torch:
            # TODO: replace with a proper error message.
            print('torch failed to import. This GAN implementation relies on torch.')
            return
        #

        self.data = []
        self.labels = []

        self.params = {}
        self.params['gen_layers'] = []          # size of generator layers, excluding output layer
        self.params['discr_layers'] = []        # size of discriminator layers, excluding output layer
        self.params['minibatch_size'] = 10      # number of real/fake data generated per iter in training
        self.params['k_inner'] = 1              # number of train iters for discriminator for every iter of generator
        self.params['n_iter'] = 100             # number of outer loops for training
        self.params['lr'] = 10.**-2             # learning rate (stepsize for optimization)
        self.params['normalize_data'] = True    # per-variable normalization. Usually want true unless you
                                                # play with the activation functions in the generator.
        self.params['verbosity'] = 0            # Set to positive value to see outputs.

        self._shift = None
        self._scale = None

        self.results = {}
        return
    #

    def fit(self,data,labels=[]):
        '''
        Args:
            - data: training data
            - labels: training labels. If not specified, assumed to be all of the same class.
        '''

        m,n = np.shape(data)

        if (m > self.params['minibatch_size']):
            self.params['minibatch_size'] = m
        #

        self.data = np.array(data, dtype=np.double)
        if len(labels)==0:
            self.labels = np.zeros(m, dtype=int)
        else:
            self.labels = np.array(labels)
        #

        labels_u = np.unique(self.labels)
        self.results['eq_tr'] = {lu: np.where(labels==lu)[0] for lu in labels_u}

        if self.params['normalize_data']:
            if self.params['verbosity']>0:
                print('GAN: normalizing data before training.')

            self._scale = {}
            self._shift = {}

            # Per-class data normalization.
            for lu,ptr in self.results['eq_tr'].items():

                self._scale[lu] = np.zeros(n, dtype=np.double)
                self._shift[lu] = np.zeros(n, dtype=np.double)

                subset = self.data[ptr]

                for j in range(n):
                    col = subset[:,j]
                    self._shift[lu][j] = np.nanmin(col)
                    col -= self._shift[lu][j]

                    self._scale[lu][j] = np.nanmax(col)
                    col /= self._scale[lu][j]

                    self.data[ptr,j] = col
                #
            #
        #

        # Push data matrix to torch.
        self.data = torch.from_numpy( self.data )

        # Set up one GAN for each unique label in the data, and
        # save the objects.
        if self.params['verbosity']>0:
            print('Initializing and training GAN for each label in the training data.')
        self.results['GANs'] = {}
        for lu in labels_u:
            subset = self.data[ self.results['eq_tr'][lu] ]

            gano = GANObject(
                subset,
                verbosity = self.params['verbosity'],
                k_inner = self.params['k_inner'],
                niter = self.params['n_iter'],
                lr = self.params['lr'],
                mb = min( len(subset), self.params['minibatch_size'] )
            )

            # EXTRA STUFF HAPPENS HERE (?)
            gano.fit()

            self.results['GANs'][lu] = gano # thing
        #

        return
    #

    def generate(self, synthlabels):
        '''
        Generate synthetic labels based on the input data.

        Inputs:
            synthlabels : list-like of labels for which corresponding synthetic
                data is requested.
        Outputs:
            synthdata : numpy array of shape len(synthlabels)-by-n, where n is
                the dimensionality of the input data.
        '''
        import numpy as np
        m,n = np.shape(self.data)
        ms = len(synthlabels)

        synthlabels_u = np.unique(synthlabels)
        synthdata = np.zeros( (ms, n), dtype=np.double )

        eq_sl = {slu: np.where(synthlabels==slu)[0] for slu in synthlabels_u}

        eq_tr = self.results['eq_tr']

        for i,slu in enumerate(synthlabels_u):
            p_s = len(eq_sl[slu])   # Number of requested synth data points in the class
            ptr = eq_sl[slu]

            synth_slu = (self.results['GANs'][slu].Gz(p_s)).detach().numpy()
            if self.params['normalize_data']:
                for j in range(n):
                    synth_slu[:,j] = self._scale[slu][j] * synth_slu[:,j] + self._shift[slu][j]
            #

            synthdata[ptr] = synth_slu
        #


        return synthdata
    #

#

class GANObject:
    def __init__(self,data,**kwargs):
        self.data = data
        m,d = np.shape(data)

        for k,v in kwargs.items():
            try:
                setattr(self,k,v)
            except:
                continue
        #

        skeys = self.__dict__.keys()

        defaults = {
        'd': d,                # dimensionality of data
        'd_noise': min(30, d),  # input layer size for generator.
        'hl': min(30,d),     # hidden layer size.
        'mb': min(m,30),         # minimum of number of data points and thirty.
        'niter': 10000,          # number of iterations
        'lr': 10**-2,            # learning rate
        'k_inner': 1            # number of discriminator iterations per generator iteration
        }

        for k,dv in defaults.items():
            if k not in skeys:
                setattr(self,k,dv)
        #

        """ Generator hidden layer variables. """

        self.Wzh = self.xavier_init(size=[self.d_noise, self.hl])
        self.bzh = Variable(torch.zeros(self.hl, dtype=torch.double), requires_grad=True)

        self.Whx = self.xavier_init(size=[self.hl, self.d])
        self.bhx = Variable(torch.zeros(self.d, dtype=torch.double), requires_grad=True)

        if self.verbosity>0:
            print('Generator initialized with layer structure %i-%i-%i'%(self.d_noise,self.hl,self.d))

        """ Discriminator hidden layer variables. """
        self.Wxh = self.xavier_init(size=[self.d, self.hl])
        self.bxh = Variable(torch.zeros(self.hl, dtype=torch.double), requires_grad=True)

        self.Why = self.xavier_init(size=[self.hl, 1])
        self.bhy = Variable(torch.zeros(1, dtype=torch.double), requires_grad=True)

        if self.verbosity>0:
            print('Discriminator initialized with layer structure %i-%i-%i'%(self.d,self.hl,1))

        return
    #

    def G(self,z):
        '''
        Evaluate the generator with the input z.
        '''
        h = nn.sigmoid(z @ self.Wzh + self.bzh.repeat(z.size(0), 1))
        X = nn.sigmoid(h @ self.Whx + self.bhx.repeat(h.size(0), 1))
        return X
    #

    def Gz(self,nsamp):
        '''
        Alias for self.G(self.noise_prior(nsamp)).
        '''
        return self.G(self.noise_prior(nsamp))
    #

    def D(self,X):
        '''
        Evaluate the discriminator with the input X.
        '''
        h = nn.sigmoid(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
        y = nn.sigmoid(h @ self.Why + self.bhy.repeat(h.size(0), 1))
        return y
    #

    def noise_prior(self,nsamp):
        '''
        This goes in to the generator.
        Unclear how the choice of the prior noise distribution would
        affect the output - likely very little.
        '''
        return torch.rand(nsamp, self.d_noise, dtype=torch.double)
    #

    def xavier_init(self,size):
        '''
        Some kind of heuristic initialization of weights for
        some of the neural net.
        '''
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return Variable(torch.randn(*size, dtype=torch.double) * xavier_stddev, requires_grad=True)

    def sample_from_data(self,size):
        '''
        Selects mb samples (uniform) randomly from self.data, without replacement.
        '''
        selections = np.random.choice(len(self.data), size=size, replace=False)
        return self.data[selections]
    #

    def fit(self):
        if self.verbosity>1:
            from scipy import stats
        #

        G_params = [self.Wzh, self.bzh, self.Whx, self.bhx]
        D_params = [self.Wxh, self.bxh, self.Why, self.bhy]
        params = G_params + D_params

        """ ===================== TRAINING ======================== """

        def reset_grad():
            for p in params:
                if p.grad is not None:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())
        #

        G_solver = optim.SGD(G_params, lr=self.lr)
        D_solver = optim.SGD(D_params, lr=self.lr)

        ones_label = Variable(torch.ones(self.mb, 1, dtype=torch.double))
        zeros_label = Variable(torch.zeros(self.mb, 1, dtype=torch.double))

        if self.verbosity>1:
            print('Beginning GAN training loop')

        for it in range(self.niter):
            # if it%10000==0 and it>0:
            #     m = int(1.2*m)  # Try increasing the sample size over training
            #                     # to force generator/discriminator to learn the variance (does this work?)
            #     ones_label = Variable(torch.ones(m, 1))
            #     zeros_label = Variable(torch.zeros(m, 1))
            #     # k_inner += 2
            # #
            for k in range(self.k_inner):
                # Sample from original data and discriminator

                z = Variable( self.noise_prior(self.mb) )
                # X, _ = mnist.train.next_batch(mb_size)
                # X = Variable(torch.from_numpy(X))
                # X = data_distribution(m)

                # m samples randomly selected from the data.
                X = self.sample_from_data(self.mb)

                # Dicriminator forward-loss-backward-update
                G_sample = self.G(z)
                D_real = self.D(X)
                D_fake = self.D(G_sample)

                D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
                D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
                D_loss = D_loss_real + D_loss_fake

                D_loss.backward()
                D_solver.step()
                # D_solver.step(eval_loss_D)    # Needed for LBFGS

                # Housekeeping - reset gradient
                reset_grad()
            #

            # Generator forward-loss-backward-update
            z = Variable( self.noise_prior(self.mb) )
            G_sample = self.G(z)
            D_fake = self.D(G_sample)

            G_loss = nn.binary_cross_entropy(D_fake, ones_label)

            G_loss.backward()
            G_solver.step()
            # G_solver.step(eval_loss_G)    # Neededfor LBFGS

            # Housekeeping - reset gradient
            reset_grad()

            # Debugging - diagnostics on real and fake data.
            if self.verbosity>1 and it%100==0:
                X_np = X.detach().numpy()
                Gs_np = G_sample.detach().numpy()
                ss_real = stats.describe(X_np)
                ss_fake = stats.describe(Gs_np)

                diff_of_means = np.linalg.norm(ss_real.mean - ss_fake.mean)/np.linalg.norm(ss_real.mean)
                diff_of_covars = np.linalg.norm(np.cov(X_np.T) - np.cov(Gs_np.T))/np.linalg.norm(np.cov(X_np))

                print('Iteration %i.\n======================'%it)
                print('Relative normed difference of means: %.3e'%diff_of_means)
                # print('Relative matrix normed difference of covariances: %.3e'%diff_of_covars)

                if it%1000==0:
                    print('Singular values:')
                    _,sr,_ = np.linalg.svd(X_np, full_matrices=False)
                    _,sf,_ = np.linalg.svd(Gs_np, full_matrices=False)
                    print(sr)
                    print(sf)
                #
                print("")
            #
        #

#


if __name__ == "__main__":
    from matplotlib import pyplot
    import numpy as np

    m = 20      # Data points per circle
    p = 1273    # Total additional data points.
    np.random.seed(57721)   # Euler-Mascheroni

    props = dict(boxstyle='square', facecolor=[0.95,0.95,0.95], edgecolor='k', alpha=0.5)

    th = 2*np.pi*np.random.rand(m)
    x = np.cos(th)
    y = np.sin(th)

    data = np.vstack((x,y)).T
    labels = np.zeros(m)

    fig,ax = pyplot.subplots(1,1, sharex=True, sharey=True)

    # data_synth_all = [[smote_oneclass(data,p,k) for k in [1,2,3,4]]]
    data_synth_all = []
    for k in [1]:
        gan = GANGenerator()
        gan.params['n_iter'] = 30000
        gan.params['verbosity'] = 2
        gan.params['minibatch_size'] = 40
        gan.params['lr'] = 10**-3
        gan.fit(data,labels)

        data_synth_all.append( gan.generate(np.zeros(p)) )
    #

    for j,k in enumerate([1]):
        ax.scatter(data_synth_all[j][:,0], data_synth_all[j][:,1], c='r', s=1)
        ax.scatter(data[:,0],data[:,1], c='k', marker=r'$\odot$', s=100)

        # ax.set_title('k=%i'%k)

        for col,lab,marker,si in [['k','Original data',r'$\odot$',100],['r','Synthetic data','.',20]]:
            ax.scatter([],[],c=col,label=lab, marker=marker, s=si)
        #
        ax.axis('equal')
    #

    fig.tight_layout()
    fig.suptitle('GAN generated data', fontsize=18)
    fig.subplots_adjust(top=0.85)

    # ##############################
    # #
    # # Example 2, with multiple classes.
    # #
    #
    # m = 200      # Data points per circle
    # p = 1273    # Total additional data points.
    # np.random.seed(57721)   # Euler-Mascheroni
    #
    # props = dict(boxstyle='square', facecolor=[0.95,0.95,0.95], edgecolor='k', alpha=0.5)
    #
    # th = 2*np.pi*np.random.rand(m)
    # x0 = np.cos(th)
    # y0 = np.sin(th)
    # th = 2*np.pi*np.random.rand(m)
    # x1 = np.cos(th) + 0.5
    # y1 = np.sin(th)
    #
    # x = np.hstack((x0,x1))
    # y = np.hstack((y0,y1))
    #
    # data = np.vstack((x,y)).T
    # labels = np.hstack( (np.zeros(m), np.ones(m)) )
    #
    # fig2,ax2 = pyplot.subplots(1,1, sharex=True, sharey=True)
    #
    # for i,k in enumerate([1]):
    #     synth_labels = np.random.choice([0,1], p)
    #     colors = ['b' if s else 'r' for s in synth_labels]
    #     gan = GANGenerator()
    #     gan.params['n_iter'] = 10000
    #     gan.params['verbosity'] = 2
    #     gan.params['minibatch_size'] = 40
    #     gan.params['lr'] = 10**-3
    #
    #     gan.fit(data,labels)
    #
    #     data_synth = gan.generate(synth_labels)
    #
    #     ax2.scatter(data_synth[:,0], data_synth[:,1], c=colors, s=1)
    #     ax2.scatter(data[:,0],data[:,1], c='k', s=40)
    #     ax2.axis('equal')
    # #
    # fig2.tight_layout()
    # fig2.suptitle('GAN generated data; two classes', fontsize=18)
    # fig2.subplots_adjust(top=0.85)


    pyplot.show(block=False)
#
