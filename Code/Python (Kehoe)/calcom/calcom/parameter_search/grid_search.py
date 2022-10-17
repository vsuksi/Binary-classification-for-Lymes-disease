from __future__ import absolute_import, division, print_function
import calcom
from ..experiment import Experiment
import inspect


'''
    Note: Multiprocessing may not be useful for classifiers that uses blas to do the most processor intesive tasks.
    Note: Does not use the newer CCExperiment class. Needs to be updated.
'''


class GridSearch(object):
    '''
    Search for the best parameters for the classifier given the dataset
    '''
    classifier = None
    pram_grid = {}
    cross_validation = None
    folds = None
    classifier_results = {}
    evaluation_metric = None
    best_params = {}
    verbosity = 1
    def __init__(self, classifier, param_grid, cross_validation="stratified_k-fold", evaluation_metric = calcom.metrics.ConfusionMatrix('bsr'), folds = 3, verbosity=1, balance_data="", use_multiprocessing=False):
        '''
        Parameters
        ----------
        classifier: classifier for which we want to do grid search
        param_grid: a dictionary of parameters with lists of discrete values.
                    Example, param_grid = {"C":[0.1,1,10], "gamma":[0.01, 0.1, 10, 1000]}
        cross_validation: type of cross-validation. i.e. leave-one-out, leave-p-out, k-fold
        folds: multi-purpose parameter. 1 for leave-1-out, integer for k-fold
        evaluation_metric: the metric used to find the best classification model for each classifier across different folds
        verbosity: integer 0, 1, or 2. If higher, more print statements
        balance_data: balance algorithm for training data

        Returns
        -------
        None
        '''
        self.classifier = classifier
        self.param_grid = param_grid
        self.cross_validation = cross_validation
        self.folds = folds
        self.evaluation_metric = evaluation_metric
        self.verbosity = verbosity
        self.balance_data = balance_data
        self.use_multiprocessing = use_multiprocessing

    #


    def mp_run_combination(self,combination):
        '''
        Helper method for multiprocessing pool. Runs on each process.
        '''
        if isinstance(self.classifier, list):
            self.classifier = self.classifier[0]

        if inspect.isclass(self.classifier):
            if hasattr(self.classifier, 'init_params'):
                clf = self.classifier()
                clf.init_params(combination)
            else:
                clf = self.classifier(**combination)
        else:
            clf = self.classifier

        exp = Experiment(data = self._data,
                        labels = self._labels,
                        classifier_list = [clf],
                        cross_validation = self.cross_validation,
                        evaluation_metric = self.evaluation_metric,
                        folds=self.folds,
                        verbosity=0,
                        balance_data=self.balance_data)
        best_classification_models = exp.run()

        # can be done because only one classifier
        for v in exp.classifier_results.values():
            mean_score = v['mean']

        if self.cross_validation == "leave-one-out":
            mean_score = v['scores']

        if self.verbosity >= 1:
            print( "%-5s : %.3f %-10s : %s" % ("Score", mean_score, "Parameters", combination) )

        return  mean_score, combination

    #


    def run(self, data, labels):
        '''
        Find the best combination of parameters given a dictionary of parameters with lists of discrete values.
        
        Inputs:
            data and labels to do the parameter search on
        Outputs:
            best combination of parameters (that are being searched) found by grid search;
        '''


        best_combination = None
        max_score = -1
        self._data = data
        self._labels = labels

        # generate all possible combination of parameters
        from sklearn.model_selection import  ParameterGrid
        all_combinations = list(ParameterGrid(self.param_grid))

        import multiprocessing
        import time
        import sys
        start_time = time.time()
        if sys.platform == 'darwin' or self.use_multiprocessing == False:
            res = []
            for combination in all_combinations:
                res += [self.mp_run_combination(combination)]
            results = iter(res)
        else:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            results = pool.imap(self.mp_run_combination, all_combinations)
            pool.close()

        for i in range(len(all_combinations)):
            mean_score,combination = next(results)
            if max_score < mean_score:
                max_score = mean_score
                best_combination = combination

        elapsed_time = time.time() - start_time
        if self.verbosity >= 1:
            print('Time Taken: {0:.2f}'.format(elapsed_time))
            print(best_combination)

        self.best_params = best_combination
        
        return best_combination
