from __future__ import absolute_import, division, print_function
import numpy as np
from .calcom import Calcom
from calcom.io import CCDataSet

from .experiment import Experiment
from .utils import generate_partitions
import copy


# There is redundancy with the original experiment class; Can be combined into one class later.
# No multiprocessing stuff
class CCExperiment(object):
    '''
    This experiment class can handle a CCDataSet.
    '''
    def __init__(self,
                classifier_list,
                ccd,
                evaluation_metric,
                classification_attr = "",
                validation_attr = "",
                cross_validation_attr = "",
                cross_validation = "leave-one-attrvalue-out",
                folds = 3,
                verbosity = 1,
                save_all_classifiers = False,
                feature_set = "",
                synth_generator = None,
                synth_strategy = "equalize",
                synth_factor = 0,
                batch_attr = "",
                idx_list = [],
                validation_idx_list = []
            ):
        '''
        Args:
            - classifier_list: list of classifiers
            - ccd: CCDataSet to run the experiment on
            - evaluation_metric: the metric used to find the best
            - classification_attr: attribute on which to do the classification
            - cross_validation_attr: attribute on which to do the cross-validation on
            - cross_validation: type of cross validation. Default: leave-one-attrvalue-out

            - batch_attr: string. Attribute on which the *entire* dataset
                should be preprocessed prior to cross-validation to remove batch effects.
                The technique used is limma, which essentially looks for the best
                least-squares model which explains the data based on which study they belong to.
                (does this boil down to accounting for class-wise means?) (Default: empty string, corresponding to no batch effect removal.)

            - folds: number of folds for cross-validation; this parameter is ignored for `leave-one-attrvalue-out` or `leave-one-out` cross-validation
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

                    'augment_majority'  : Synth data is generated for the *majority* class
                                            by a factor (1+self.synth_factor).

            - synth_factor: nonnegative float. If synth_strategy=='equalize', this determines
                the factor multiplying the size of the majority class for the
                target amount of total data. (default: 0)

            - idx_list: restrict available data to a particular subset.
            - validation_idx_list: A set of indices indicating location of
                a SEQUESTERED validation set.
                An error is thrown upon initialization if there is
                any overlap between this list and idx_list.
        '''

        self.classifier_list = classifier_list
        self.ccd = ccd
        self.evaluation_metric = evaluation_metric
        self.classification_attr = classification_attr
        self.validation_attr = validation_attr
        self.cross_validation_attr = cross_validation_attr
        self.cross_validation = cross_validation
        self.folds = folds

        self.verbosity = verbosity
        self.classifier_results = dict()
        self.best_classifiers = dict()
        self.save_all_classifiers = save_all_classifiers

        if len(np.intersect1d(idx_list, validation_idx_list))>0:
            raise Exception("The input pointer set idx_list and validation_idx_list must not overlap.")

        self.idx_list = idx_list
        self.validation_idx_list = validation_idx_list
        self.feature_set = feature_set

        self.synth_generator = synth_generator
        self.synth_strategy = synth_strategy
        self.synth_factor = synth_factor

        self.batch_attr = batch_attr

        if len(self.idx_list)==0:
            self.idx_list = np.arange(len(ccd.data))
        #
    #

    def run(self):
        '''
        run the classifier(s) with specified cross-validation and synthetic data generator
        '''
        from calcom.metrics import ConfusionMatrix

        ccd = self.ccd
        verbosity = self.verbosity

        data = ccd.generate_data_matrix(feature_set=self.feature_set, idx_list=self.idx_list)
        labels = ccd.generate_labels(self.classification_attr, idx_list=self.idx_list)

        self.validation_data = ccd.generate_data_matrix(feature_set=self.feature_set, idx_list=self.validation_idx_list)
        self.validation_labels = ccd.generate_labels(self.validation_attr, idx_list=self.validation_idx_list)

        if len(self.batch_attr)>0:
            data = self.normalize_data(data)
        #


        if data.shape[0] < self.folds:
            raise ValueError("Too many folds")
        elif ( (data.shape[0] == self.folds) and (self.cross_validation != "leave-one-out") ):
            print("Number of folds is same as number of data points; switching to leave-one-out cross-validation.")
            self.cross_validation = "leave-one-out"
        #

        if (verbosity>=1):
            print("Partitioning data for cross-validation... ",end="")
        #

        if self.cross_validation=='leave-one-attrvalue-out':
            _label_partitions = generate_partitions(labels=labels,
                                                method=self.cross_validation,
                                                nfolds=self.folds,
                                                attrvalues=ccd.generate_labels(self.cross_validation_attr, idx_list=self.idx_list))
        else:
            # Calling by the cross_validation_attr doesn't make sense.
            _label_partitions = generate_partitions(labels=labels,
                                                method=self.cross_validation,
                                                nfolds=self.folds)
        #

        self._label_partitions = _label_partitions

        if (verbosity>=1):
            print("done.")
        #

        if (verbosity>=1 and type(self.cross_validation)==str):
            print("Performing %s cross-validation... \n"%self.cross_validation)
        #


        if self.cross_validation=='leave-one-attrvalue-out' or self.cross_validation=='leave-one-out':

            for count, classifier in enumerate(self.classifier_list):
                # clf_metrics = np.zeros(self.folds)
                cname = classifier.__class__.__name__ + "_" + str(count)
                clfs = []

                labels_pred = np.array(labels)
                for i,partition in enumerate(_label_partitions):
                    clf_copy = copy.deepcopy(classifier)

                    preds, test_idx, test_labels, clf = self.split_and_train(partition, data, labels, clf_copy)

                    labels_pred[test_idx] = preds


                    # clf_metrics[i] = self.evaluation_metric.evaluate(test_labels,preds)
                    clfs = clfs + [clf]

                    if (self.verbosity>=1):
                        # The zfill should really be done based on number of digits in len(label_partitions).
                        # The "\r" sends the cursor to the beginning of the same line to rewrite previous
                        # iteration printed.
                        print("%-30s: %3.0i of %i"%(cname,i+1,len(_label_partitions)), end="\r")
                    #
                #
                if (self.verbosity>=1):
                    print("")
                #

                cf = ConfusionMatrix()
                _ = cf.evaluate(labels,labels_pred)
                self.classifier_results[cname] = {
                    'true_labels' : labels,
                    'pred_labels' : labels_pred,
                    'confmat' : cf,
                    'scores' : self.evaluation_metric.evaluate(labels, labels_pred),
                    'mean'   : None,
                    'min'    : None,
                    'max'    : None,
                    'std'    : None
                }

                # For the same reason it also doesn't really make sense to talk about
                # a "best" classifier, since (ideally) a large percentage
                # of the classifiers will succeed on the test datum.
                if self.save_all_classifiers:
                    self.best_classifiers[classifier.__class__.__name__] = clfs
                else:
                    # Currently, save a random choice of classifiers,
                    # since there's no real distinguishing between them in leave-one-out.
                    self.best_classifiers[classifier.__class__.__name__] = clfs[0] #np.random.choice(clfs)
                #
            #

            if (self.verbosity>=1):
                # Print summary statistics.
                train_idx,test_idx = _label_partitions[0]
                print("")
                # print("%s" % "-"*50)
                # print("")
                print("%-20s %s %20s"%( ("-"*20), "Experiment complete.", ("-"*20) ) )
                print("")
                print("%-30s : %6i"%("Total number of samples",len(train_idx)+len(test_idx)))
                print("%-30s"%(self.cross_validation +" cross validation performed."))
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
            #
        else:
            for count, classifier in enumerate(self.classifier_list):
                clf_metrics = np.zeros(self.folds)
                cname = classifier.__class__.__name__ + "_" + str(count)
                clfs = []

                pred_labels_set = []

                # how did this get in here
                # true_labels_set = [partition[1] for partition in _label_partitions]
                #
                # This is on me, but (a) this code rarely gets accessed since
                # CCExperiment is mainly used for one-attrvalue-out;
                # (b) this is only used for construction of confusion matrices
                # for every fold of the cross-validation.
                true_labels_set = [ labels[partition[1]] for partition in _label_partitions]

                for i,partition in enumerate(_label_partitions):

                    preds, test_idx, test_labels, clf = self.split_and_train(partition, data, labels, classifier)

                    pred_labels_set.append( preds )

                    #print(test_labels,preds)
                    clf_metrics[i] = self.evaluation_metric.evaluate(test_labels,preds)
                    clfs = clfs + [clf]

                    if (self.verbosity>=1):
                        # The zfill should really be done based on number of digits in len(label_partitions).
                        # The "\r" sends the cursor to the beginning of the same line to rewrite previous
                        # iteration printed.
                        print("%-30s: %3.0i of %i"%(cname,i+1,len(_label_partitions)), end="\r")
                        #pass
                    #
                #

                if (self.verbosity>=1):
                    print("")
                #

#                import pdb
#                pdb.set_trace()

                confmats = []
                for true,pred in zip(true_labels_set, pred_labels_set):
                    cf = ConfusionMatrix()
                    _ = cf.evaluate(true,pred)
                    confmats.append(cf)
                #

                self.classifier_results[cname] = {
                    'true_labels' : true_labels_set,
                    'pred_labels' : pred_labels_set,
                    'confmat' : confmats,
                    'scores' : clf_metrics,
                    'mean' : np.mean(clf_metrics),
                    'min' : min(clf_metrics),
                    'max' : max(clf_metrics),
                    'std' : np.std(clf_metrics)
                }

                if self.save_all_classifiers:
                    self.best_classifiers[classifier.__class__.__name__] = clfs
                else:
                    self.best_classifiers[classifier.__class__.__name__] = clfs[np.argmax(clf_metrics)]
                #
            #

            if (verbosity>=1):
                # Print summary statistics.
                train_idx,test_idx = _label_partitions[0]
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
                    print( ("%-10s : %.3f \xB1 %.3f") % ("Mean",self.classifier_results[key]['mean'],self.classifier_results[key]['std']) )
                    print( "%-10s : %.3f" % ("Maximum",self.classifier_results[key]['max']) )
                    print( "%-10s : %.3f" % ("Minimum",self.classifier_results[key]['min']) )

                    print("%s" % ("-"*20))
                    print('\n')
                #
            #
        #

        return self.best_classifiers

    def normalize_data(self, data):
        from calcom.utils import limma, get_linear_batch_shifts

        #check if there's a sequestered validation data
        ccd = self.ccd

        if self.validation_data.shape[0] == 0:
            #if not, apply limma on just the training data
            batch_labels = ccd.generate_labels(self.batch_attr, idx_list=self.idx_list)

            # store the transforms per-batch 
            self.batch_shifts = get_linear_batch_shifts(data, batch_labels)

            #return normalized training data
            normalized_training_data = limma(data, batch_labels)

            return normalized_training_data
        else:
            #otherwise, concatenate training and validation data
            all_data = np.vstack((data, self.validation_data))

            #apply limma on concatenated data

            all_idx = list(self.idx_list) + list(self.validation_idx_list)

            batch_labels = ccd.generate_labels(self.batch_attr, idx_list=all_idx)

            # store the transforms per-batch
            self.batch_shifts = get_linear_batch_shifts(data,batch_labels)

            all_data = limma(all_data, batch_labels)

            #split it back into training and validation data
            normalized_training_data = all_data[: len(self.idx_list), :]
            self.validation_data = all_data[len(self.idx_list):, :]

            #return normalized training data
            return normalized_training_data


    def split_and_train(self, partition, data, labels, classifier):
        train_idx,test_idx = partition

        train_data = data[train_idx,:]
        train_labels = labels[train_idx]
        test_data = data[test_idx,:]
        test_labels = labels[test_idx]

        # Balance training data based on inputs.
        # print(np.shape(train_data), np.shape(train_labels), max([sum(train_labels==tru) for tru in np.unique(train_labels)]))
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
        return preds, test_idx, test_labels, clf
    #
#
