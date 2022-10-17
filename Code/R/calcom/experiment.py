from __future__ import absolute_import, division, print_function
# import numpy as np
# from .calcom import Calcom

# from .utils import generate_partitions # Careful, this is a relative import.
# import copy

class Experiment(object):
    data = None
    labels = None
    classifier_list = None
    cross_validation = None
    folds = None
    classifier_results = {}
    evaluation_metric = None
    best_classifiers = {}

    #

    def __init__(self,data, labels, classifier_list, cross_validation, evaluation_metric, folds=3, verbosity=1, save_all_classifiers=False, use_multiprocessing=False, synth_generator=None, synth_strategy='equalize', synth_factor=0, batch_labels=[]):
        '''
        Args:
            - data: list of features
            - label: list of labels
            - classifier_list: list of classifiers
            - cross_validation: type of cross-validation. i.e. leave-one-out, leave-p-out, k-fold
                alternatively, pass the partioning
            - evaluation_metric: the metric used to find the best
              classification model for each classifier across different folds
            - folds: multi-purpose parameter. k-1 for leave-1-out, integer for
              k-fold, ignored for leave-one-out.
            - verbosity: integer 0, 1, or 2. If higher, more print statements.

            - synth_generator: balance algorithm for training data; one of the objects
                from calcom.synthdata If none supplied, no augmentation is done.
            - synth_strategy: string. If a synthetic data generator is specified,
                this specifies the strategy for generating additional data:
                    'equalize'  : Synth data is generated for minority classes sufficient
                                    to equalize to the size of the majority class(es). (Default)

                                    If synth_factor is nonzero, additional data is generated
                                    for all classes in addition to the amount needed to
                                    equalize. For example, if the majority class has
                                    30 data points, and synth_factor==1, then all classes
                                    are augmented with an additional 30 data points after
                                    equalization.
                    'augment_majority'  : Instead of the standard technique of
                                    augmenting the minority class of the data, instead
                                    augment the majority. Why? Don't know. An additional
                                    (1+self.synth_factor) amount of synthetic data is
                                    generated.

            - batch_labels: list-like. Labels on which *all* data
                should be preprocessed prior to cross-validation to remove batch effects.
                The technique used is limma, which essentially looks for the best
                least-squares model which explains the data based on which study they belong to.
                (Default: no batch processing.)

            - synth_factor: nonnegative float. If synth_strategy=='equalize', this determines
                the factor multiplying the size of the majority class for the
                target amount of total data. (default: 0)

            - use_multiprocessing: whether to use the python multiprocessing package for parallelization (default: False)
            - multiprocessing_procs: how many processes to begin with multiprocessing (default: multiprocessing.cpu_count())
        '''
        from numpy import array

        self.data = array(data)
        self.labels = array(labels)
        self.classifier_list = classifier_list
        self.cross_validation = cross_validation
        self.folds = folds
        self.classifier_results = {}
        self.evaluation_metric = evaluation_metric
        self.best_classifiers = {}
        self.verbosity = verbosity
        self.synth_generator = synth_generator
        self.synth_strategy = synth_strategy
        self.synth_factor = synth_factor

        self.batch_labels = batch_labels

        self.save_all_classifiers = save_all_classifiers
        self.use_multiprocessing = use_multiprocessing
    #

    def mp_clf_process(self, classifier):
        import numpy as np
        import copy
        from calcom.io import CCList
        from calcom.metrics import ConfusionMatrix

        count, classifier = classifier
        clf_metrics = np.zeros(self.folds)
        clf_preds = CCList([])
        cname = classifier.__class__.__name__ + "_" + str(count)
        clfs = []

        true_labels_set = [np.array(self.labels)[partition[1]] for partition in self._label_partitions]
        pred_labels_set = []

        for i,partition in enumerate(self._label_partitions):

            train_idx,test_idx = partition

            train_data = self.data[train_idx,:]
            train_labels = self.labels[train_idx]
            test_data = self.data[test_idx,:]
            test_labels = self.labels[test_idx]

            # Balance training data based on inputs.

            if (self.synth_strategy == 'equalize') and (self.synth_generator):
                tr_unique = np.unique(train_labels)
                tr_sizes = [sum(train_labels==tru) for tru in tr_unique]
                maj_size = np.max(tr_sizes)
                synth_labels = [[tr_unique[i] for j in range(maj_size-tr_s)] for i,tr_s in enumerate(tr_sizes)]
                synth_labels = np.hstack(synth_labels) # This might cause typing issues at some point; hstack recasts as it pleases.

                if self.synth_factor>0:
                    addl = int(maj_size*self.synth_factor)
                    for tr_u in tr_unique:
                        synth_labels = np.append(synth_labels, [tr_u for i in range(addl)])
                    #
                #
                self.synth_generator.fit(train_data, train_labels)
                synth_data = self.synth_generator.generate(synth_labels)

                train_data = np.vstack( (train_data, synth_data) )
                train_labels = np.hstack( (train_labels, synth_labels) )
            elif (self.synth_strategy == 'augment_majority') and (self.synth_generator):
                # Augment the *majority class* only. Counter-intuitive, isn't it?
                tr_unique = np.unique(train_labels)
                tr_sizes = [sum(train_labels==tru) for tru in tr_unique]
                maj_size = np.max(tr_sizes)
                maj_label = tr_unique[np.argmax(tr_sizes)] # TODO: account for multiple majority classes.

                addl = int(maj_size*(1+self.synth_factor))

                synth_labels = np.array([maj_label for i in range(addl)])

                self.synth_generator.fit(train_data, train_labels)
                synth_data = self.synth_generator.generate(synth_labels)

                train_data = np.vstack( (train_data, synth_data) )
                train_labels = np.hstack( (train_labels, synth_labels) )
            #

            clf = copy.deepcopy(classifier)
            clf.fit(train_data,train_labels)
            preds = clf.predict(test_data)

            clf_metrics[i] = self.evaluation_metric.evaluate(test_labels,preds)
            clf_preds += CCList([preds])
            clfs = clfs + CCList([clf])

            if (self.verbosity>=1):
                # The zfill should really be done based on number of digits in len(label_partitions).
                # The "\r" sends the cursor to the beginning of the same line to rewrite previous
                # iteration printed.
                print("%-30s: %3.0i of %i"%(cname,i+1,len(self._label_partitions)), end="\r")
                #pass
            #
        #

        if (self.verbosity>=1):
            print("")
        #

        confmats = []
        for true,pred in zip(true_labels_set,clf_preds):
            cf = ConfusionMatrix()
            _ = cf.evaluate(true,pred)
            confmats.append(cf)
        #

        return cname, clfs, clf_metrics, {
                    'true_labels' : true_labels_set,
                    'pred_labels' : clf_preds,
                    'confmat' : confmats,
                    'scores' : clf_metrics,
                    'mean' : np.mean(clf_metrics),
                    'min' : min(clf_metrics),
                    'max' : max(clf_metrics),
                    'std' : np.std(clf_metrics),
                }

    def mp_leave_one_out(self,classifier):
        import numpy as np
        from calcom.io import CCList
        import copy
        from calcom.metrics import ConfusionMatrix

        count,classifier = classifier
        # clf_metrics = np.zeros(self.folds)
        cname = classifier.__class__.__name__ + "_" + str(count)
        clfs = []

        labels_pred = np.array(self.labels)
        for i,partition in enumerate(self._label_partitions):

            train_idx,test_idx = partition

            train_data = self.data[train_idx,:]
            train_labels = self.labels[train_idx]
            test_data = self.data[test_idx,:]
            test_labels = self.labels[test_idx]


            if (self.synth_strategy == 'equalize') and (self.synth_generator):
                tr_unique = np.unique(train_labels)
                tr_sizes = [sum(train_labels==tru) for tru in tr_unique]
                maj_size = np.max(tr_sizes)
                synth_labels = [[tr_unique[i] for j in range(maj_size-tr_s)] for i,tr_s in enumerate(tr_sizes)]
                synth_labels = np.hstack(synth_labels) # This might cause typing issues at some point; hstack recasts as it pleases.

                if self.synth_factor>0:
                    addl = int(maj_size*self.synth_factor)
                    for tr_u in tr_unique:
                        synth_labels = np.append(synth_labels, [tr_u for i in range(addl)])
                    #
                #
                self.synth_generator.fit(train_data, train_labels)
                synth_data = self.synth_generator.generate(synth_labels)

                train_data = np.vstack( (train_data, synth_data) )
                train_labels = np.hstack( (train_labels, synth_labels) )
            elif (self.synth_strategy == 'augment_majority') and (self.synth_generator):
                # Augment the *majority class* only. Counter-intuitive, isn't it?
                tr_unique = np.unique(train_labels)
                tr_sizes = [sum(train_labels==tru) for tru in tr_unique]
                maj_size = np.max(tr_sizes)
                maj_label = tr_unique[np.argmax(tr_sizes)] # TODO: account for multiple majority classes.

                addl = int(maj_size*(1+self.synth_factor))

                synth_labels = np.array([maj_label for i in range(addl)])

                self.synth_generator.fit(train_data, train_labels)
                synth_data = self.synth_generator.generate(synth_labels)

                train_data = np.vstack( (train_data, synth_data) )
                train_labels = np.hstack( (train_labels, synth_labels) )
            #

            clf = copy.deepcopy(classifier)
            clf.fit(train_data,train_labels)
            preds = clf.predict(test_data)
            labels_pred[test_idx] = preds[0]


            # clf_metrics[i] = self.evaluation_metric.evaluate(test_labels,preds)
            clfs = clfs + [clf]

            if (self.verbosity>=1):
                # The zfill should really be done based on number of digits in len(label_partitions).
                # The "\r" sends the cursor to the beginning of the same line to rewrite previous
                # iteration printed.
                print("%-30s: %3.0i of %i"%(cname,i+1,len(self._label_partitions)), end="\r")
            #
        #

        if (self.verbosity>=1):
            print("")
        #

        cf = ConfusionMatrix()
        _ = cf.evaluate(self.labels, labels_pred)

        return cname,clfs,{
            'true_labels' : self.labels,
            'pred_labels' : labels_pred,
            'confmat' : cf,
            'scores' : self.evaluation_metric.evaluate(self.labels, labels_pred),
            'mean'   : None,
            'min'    : None,
            'max'    : None,
            'std'    : None
        }
    #



    def run(self):
        verbosity = self.verbosity

        if self.data.shape[0] < self.folds:
            raise ValueError("Too many folds")
        elif ( (self.data.shape[0] == self.folds) and (self.cross_validation != "leave-one-out") ):
            print("Number of folds is same as number of data points; switching to leave-one-out cross-validation.")
            self.cross_validation = "leave-one-out"
        #

        if (verbosity>=1):
            print("Partitioning data for cross-validation... ",end="")
        #

        if len(self.batch_labels)>0:
            from calcom.utils import limma, get_linear_batch_shifts

            # compute shifts before applying the transform to data.
            self.batch_map = get_linear_batch_shifts(self.data, self.batch_labels)

            self.data = limma(self.data, self.batch_labels)
        #

        # label_partitions = self.partition(self.labels, method=self.cross_validation, nfolds=self.folds)
        if type(self.cross_validation) == str:
            from calcom.utils import generate_partitions

            self._label_partitions = generate_partitions(self.labels, method=self.cross_validation, nfolds=self.folds)
        else:
            self._label_partitions = self.cross_validation
            self.folds = len(self._label_partitions)

        if (verbosity>=1):
            print("done.")
        #

        if (verbosity>=1 and type(self.cross_validation)==str):
            print("Performing %s cross-validation... \n"%self.cross_validation)
        #

        # Keep a record of each classifier's results
        # for every fold to calculate mean, std, max, min score afterward.
        if self.cross_validation=='leave-one-out':
            # Things need to be handled differently here;
            # there is only one metric that can be measured, since
            # it doesn't make sense to ask for anything but
            # accuracy for a test set of a single data point.
            import multiprocessing
            import time
            import sys
            start_time = time.time()
            if len(self.classifier_list) > 1:
                if sys.platform == 'darwin' or self.use_multiprocessing == False:
                    res = []
                    for count,classifier in enumerate(self.classifier_list):
                        res += [self.mp_leave_one_out((count,classifier))]
                    results = iter(res)
                else:
                    pool = multiprocessing.Pool(multiprocessing.cpu_count())
                    results = pool.imap(self.mp_leave_one_out, enumerate(self.classifier_list))
                    pool.close()
            else:
                results = iter([self.mp_leave_one_out((0,self.classifier_list[0]))])

            for classifier in self.classifier_list:
                cname, clfs, res = next(results)
                # Doesn't make sense to talk about aggregate statistics
                # in the way we do with k-fold, since we
                # only have one classification for each point.
                self.classifier_results[cname] = res

                # For the same reason it also doesn't really make sense to talk about
                # a "best" classifier, since (ideally) a large percentage
                # of the classifiers will succeed on the test datum.
                if self.save_all_classifiers:
                    self.best_classifiers[classifier.__class__.__name__ ] = clfs
                else:
                    # Currently, save a random choice of classifiers,
                    # since there's no real distinguishing between them in leave-one-out.
                    self.best_classifiers[classifier.__class__.__name__ ] = clfs[0] #np.random.choice(clfs)
                #
                self.classifier_results[cname]



            elapsed_time = time.time() - start_time
            if self.verbosity >= 2:
                print('Time Taken: {0:.2f}'.format(elapsed_time))



            if (verbosity>=1):
                # Print summary statistics.
                train_idx,test_idx = self._label_partitions[0]
                print("")
                # print("%s" % "-"*50)
                # print("")
                print("%-20s %s %20s"%( ("-"*20), "Experiment complete.", ("-"*20) ) )
                print("")
                print("%-30s : %6i"%("Total number of samples",len(train_idx)+len(test_idx)))
                print("%-30s"%"Leave-one-out cross validation performed.")
                print("")
                print("%s" % "-"*50)

                for key in self.classifier_results.keys():
                    # return_measure ONLY WORKS FOR CONFUSION MATRIX RIGHT NOW
                    print("%s statistics for %s:\n" % (self.evaluation_metric.return_measure,key) )
                    # print( ("%-10s : %.3f \xB1 %.3f") % ("Mean",self.classifier_results[key]['mean'],self.classifier_results[key]['std']) )
                    print( ("%-10s : %.3f") % (self.evaluation_metric.return_measure, self.classifier_results[key]['scores']) )

                    print("%s" % ("-"*20))
                    print('\n')
                #
        else:

            import multiprocessing
            import time
            import sys
            from numpy import argmax

            start_time = time.time()
            if len(self.classifier_list) > 1:
                if sys.platform == 'darwin' or self.use_multiprocessing == False:
                    res = []
                    for count,classifier in enumerate(self.classifier_list):
                        res += [self.mp_clf_process((count,classifier))]
                    results = iter(res)
                else:
                    pool = multiprocessing.Pool(multiprocessing.cpu_count())
                    results = pool.imap(self.mp_clf_process, enumerate(self.classifier_list))
                    pool.close()
            else:
                results = iter([self.mp_clf_process((0,self.classifier_list[0]))])

            for classifier in self.classifier_list:
                cname, clfs, clf_metrics, res = next(results)

                self.classifier_results[cname] = res

                if self.save_all_classifiers:
                    self.best_classifiers[classifier.__class__.__name__ ] = clfs
                else:
                    self.best_classifiers[classifier.__class__.__name__ ] = clfs[argmax(clf_metrics)]
                #


            #

            elapsed_time = time.time() - start_time
            if self.verbosity >= 2:
                print('Time Taken: {0:.2f}'.format(elapsed_time))

            if (verbosity>=1):
                # Print summary statistics.
                train_idx,test_idx = self._label_partitions[0]
                print("")
                # print("%s" % "-"*50)
                # print("")
                print("%-20s %s %20s"%( ("-"*20), "Experiment complete.", ("-"*20) ) )
                print("")
                print("%-30s : %6i"%("Total number of samples",len(train_idx)+len(test_idx)))
                print("%-30s : %6i"%("Approximate training samples",len(train_idx)))
                print("%-30s : %6i"%("Approximate testing samples",len(test_idx)))
                print("")
                print("%s" % "-"*50)

                for key in self.classifier_results.keys():
                    # return_measure ONLY WORKS FOR CONFUSION MATRIX RIGHT NOW
                    print("%s statistics for %s:\n" % (self.evaluation_metric.return_measure,key) )
                    print( ("%-10s : %.3f \xB1 %.3f") % ("Mean",self.classifier_results[key]['mean'], 2*self.classifier_results[key]['std']) )
                    print( "%-10s : %.3f" % ("Maximum",self.classifier_results[key]['max']) )
                    print( "%-10s : %.3f" % ("Minimum",self.classifier_results[key]['min']) )

                    print("%s" % ("-"*20))
                    print('\n')
                #
            #
        #

        #

        return self.best_classifiers
    #
