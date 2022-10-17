if __name__=="__main__":
    '''
    Demonstrating use of our built-in Bayesian scheme to optimize
    hyperparameters for a given cross-validation scheme.


    Note that this is likely out of date (as of February 4, 2019)
    '''
    import calcom
    from sklearn.svm import SVC

    from calcom import BayesianOptimization
    from calcom import Experiment
    from calcom.classifiers import SSVMClassifier

    # # Load data and labels
    # ccom = calcom.Calcom()
    # data,labels = ccom.load_data('../data/final/duke_metab_29h_logged_scaled.csv',shuffle=False)

    bsr = calcom.metrics.ConfusionMatrix('bsr')

    def svccv(**args):
        clf = SSVMClassifier()
        clf.init_params(args)
        #clf = SVC(kernel='rbf',**args)
        expObj = Experiment(data = data,
            labels = labels,
            classifier_list = [ clf ] ,
            cross_validation = "leave-one-out",
            evaluation_metric = bsr,
            folds=3,
            verbosity=0)
        expObj.run()
        for v in expObj.classifier_results.values():
            mean_score = v['scores']

        return mean_score

    import math
    f = lambda C, gamma: math.exp(-C**2 * gamma**3 + 3*gamma + C)



    gp_params = {"alpha": 1e-5}

    svcBO = BayesianOptimization(svccv,
        {'C': (0.001, 10), 'gamma': (0.0001, 0.1)})

    svcBO.explore({'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]})

    svcBO.maximize(n_iter=50, **gp_params)

    print('-' * 53)
    #rfcBO.maximize(n_iter=10, **gp_params)

    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    #print('RFC: %f' % rfcBO.res['max']['max_val'])
