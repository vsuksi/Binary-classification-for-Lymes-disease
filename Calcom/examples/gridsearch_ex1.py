if __name__ == "__main__":
    '''
    Example usage of the grid search code we have
    built in. The purpose is to find the optimal set of
    hyperparameters over all possible combinations specified.

    This is a brute force technique, in contrast to the
    Bayesian optimization scheme. There is a potential
    tradeoff in using one over the other depending on
    the details of your classifier and the properties of
    the hyperparameter search space.
    '''
    import calcom
    # from calcom.metrics import Accuracy, ConfusionMatrix
    from calcom import GridSearch
    from calcom.classifiers import SSVMClassifier

    # ccom = Calcom()

    # data,labels = ccom.load_data('../data/CS29.csv',shuffle=False)

    bsr = ConfusionMatrix('bsr')
    param_grid = {
        'C' : [1, 10, 100, 10000],
        'TOL' : [0.001, 0.01]
        }

    gridSearch = GridSearch(
                     classifier = SSVMClassifier ,
                     param_grid = param_grid,
                     cross_validation = "stratified_k-fold",
                     evaluation_metric = bsr,
                     balance_data="smote:5",
                     verbosity=1,
                     folds=5)

    best_combination = gridSearch.run(data,labels)
    print("Best set of params: ", best_combination)
    #
