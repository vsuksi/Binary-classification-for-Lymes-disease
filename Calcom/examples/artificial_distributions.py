
if __name__=="__main__":
    import numpy as np

    def reset_seed(seed=31415926):
        np.random.seed(seed)
        return
    #

    def point_clouds_2d(npoints=200,mu=[-2,-4]):

        data = np.zeros( (npoints,2) )
        labels = np.zeros(npoints)

        for i in range(npoints):
            data[i,0] = np.random.randn() + np.mod(i,2)*mu[0]
            data[i,1] = np.random.randn() + np.mod(i,2)*mu[1]
            labels[i] = np.mod(i,2)
        #
        return data,labels
    #

    def point_clouds_2d_triple(npoints=300,mus=[[0,0],[-2,0],[2,0]],balanced=True):

        data = np.zeros( (npoints,2) )
        labels = np.zeros(npoints)


        for i in range(npoints):
            if balanced:
                j = np.mod(i,4)
                if (j==3): j=0
            else:
                j = np.mod(i,3)
            #

            data[i,0] = np.random.randn() + mus[j][0]
            data[i,1] = np.random.randn() + mus[j][1]

            labels[i] = int(j>0)
        #
        return data,labels
    #

    def grid_2d(bounds=[-1,1,-1,1],ng=41):
        xvs = np.linspace(bounds[0],bounds[1],ng)
        yvs = np.linspace(bounds[2],bounds[3],ng)
        xg,yg = np.meshgrid(xvs,yvs)
        xgf = np.concatenate(xg)
        ygf = np.concatenate(yg)

        data = np.array([xgf,ygf]).T

        return xg,yg,data
    #
