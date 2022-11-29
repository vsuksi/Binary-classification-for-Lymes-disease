from __future__ import absolute_import, division, print_function
from calcom.visualizers._abstractvisualizer import AbstractVisualizer

class NMFVisualizer(AbstractVisualizer):

    def __init__(self):
        '''
        Uses same params as the NFMClassifier
        Additionally use 'dim' as dimention
        '''
        from calcom.classifiers import NMFClassifier

        self.clf = NMFClassifier()
        self.params = self.clf.params
        self.params['dim'] = 2
    #

    # def project(self,data,labels,readable_label_map={}, dim=None):
    def project(self,data, **kwargs):
        '''
        Inputs:
            data: data array, n-by-m, where n is the number of observations
                and m is the dimensionality of the data.

        Optional inputs:
            dim: dimensionality of projection; overwrites self.params['dim']

        Outputs:
            coordinate array, n-by-d, where d is the number of dimensions
                of the projection.

        '''
        import numpy as np

        self.params['dim'] = kwargs.get('dim', self.params['dim'])
        self.params['r'] = self.params['dim']

        self.clf.fit(data)
        #self.labels = self.clf.predict(data)
        # self.labels = labels
        # self.readable_label_map = readable_label_map

        # Get clusters and project the data.
        # Note this does a lot of the same work that self.predict() does.

        Q,R = self.clf.params['QR']

        datacopy = data.T
        dotprod = np.dot(Q.T,datacopy)

        projections = np.linalg.solve( R, dotprod )
        proj_data = projections.T

        # Indexes of "best" dimensions to visualize with.
        # For now let's look at one-norm variation from the means.
        means = np.mean(proj_data,axis=0)
        merp = np.array( [ row - means for row in proj_data ] )
        merp = np.linalg.norm(merp, ord=1, axis=0)
        di = np.sort( np.argsort( merp )[-dim:] )

        return proj_data[:,di]
    #

    # def visualize(self,coords):
    #     '''
    #     Input: n by d array of (projected) data.
    #     Output: pyplot figure and axes handles with the scatterplotted data.
    #
    #     Any extra arguments are passed to matplotlib.pyplot.scatter().
    #
    #     '''
    #     d = self.params['dim']
    #     if (d==2 or d==3):
    #         plotter.scatterTrainTest(coords,self.labels,None,None,readable_label_map=self.readable_label_map,title="PCA Visualizer", dim=d)
    #     else:
    #         print('Error: only d=2 and d=3 are supported for scatterplotting PCA')
    #         return None,None
    # #

    def visualize(self,coords,labels=None,label_map=None):
        '''
        Inputs:
            coords: n-by-d array of projected coordinates, with d=2 or 3.
                Automatically handles making a 2d/3d plot based on the value of d.
        Optional inputs:
            labels: (n,) array of labels.
            label_map: dictionary mapping labels to plaintext descriptions.
                If not specified, the values in labels are used.
            show: Boolean. Whether to show the plot immediately. (Default: True)
        Outputs:
            fig,ax: pyplot figure/axis pair of the resulting plot, as generated
                from fig,ax = pyplot.subplots(1,1).
        '''

        raise NotImplementedError('Visualization tools under renovation.')
        return
    #

#
