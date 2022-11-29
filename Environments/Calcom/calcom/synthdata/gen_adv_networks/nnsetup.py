import numpy as np

class Blerp:
    pass

class NN:
    '''
    A basic object following nnsetup.m from Ariel.
    '''

    def __init__(self,architecture):

        self.size = architecture    # Layer structure
        self.n = len(self.size)     # Number of layers

        self.layers = []
        for i in range(1,self.n):
            self.layers.append( Blerp() )
            self.layers[i-1].w = 0.1*np.random.randn(self.size[i-1], self.size[i])
            self.layers[i-1].b = np.zeros((1,self.size[i]))
        #

        self.layers.append( Blerp() )   # Placeholder?
        self.layers[-1].b = np.zeros( (1,self.size[-1]) )
    #

    def nngradient(self,learning_rate):
        '''
        Computes a gradient for all the layers.

        Inputs:
            learning_rate   - scalar
        Outputs:
            self            - updated object.
        '''
        for i in range(self.n-1):
            dw = self.layers[i].dw
            db = self.layers[i].db
            self.layers[i].w -= learning_rate*dw
            self.layers[i].b -= learning_rate*db
        #
        return self
    #

    def nngradient2(self,learning_rate, dw1, db1, dw2, db2):
        '''
        Computes a gradient for the first two layers only (?)

        Inputs:
            learning_rate, dw1, db1, dw2, db2   - scalars
        Outputs:
            self    - updated object.
        '''
        self.layers[0].w -= learning_rate*dw1
        self.layers[0].b -= learning_rate*db1
        self.layers[1].w -= learning_rate*dw2
        self.layers[1].b -= learning_rate*db2
        return self
    #

    def nnforward(self,x):
        '''
        forward propagation (?)

        Inputs:
            x       - input data point.
        Outputs:
            self    - updated object
        '''
        import numpy as np
        from numpy import matlib
        self.layers[0].a = x

        for i in range(1,self.n):
            a = self.layers[i-1].a
            w = self.layers[i-1].w
            b = self.layers[i-1].b

            self.layers[i].a = np.dot(a,w) + matlib.repmat(b, np.shape(a)[0], 1)
            self.layers[i].a = self.sigmoid(self.layers[i].a)
        #
        return self
    #

    def sigmoid(self,x):
        '''
        returns 1/(1+exp(-x)). No input checking is done.
        '''
        return 1/(1+np.exp(-x))
    #
#
