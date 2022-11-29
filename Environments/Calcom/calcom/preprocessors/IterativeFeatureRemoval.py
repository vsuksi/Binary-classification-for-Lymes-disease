# from __future__ import absolute_import, division, print_function
from calcom.preprocessors import AbstractPreprocessor
import numpy as np

class IFR(AbstractPreprocessor):

    def __init__(self):
        '''
        Default parameters:

        'cutoff'    : Threshold for the BSR (balanced success rate) to halt the process. (default: 0.75)
        'jumpratio' : The relative drop in the magnitude of coefficients in weight vector to identify numerically zero weights (default: 100)
        'verbosity' : Determines verbosity of print statments; 0 for no output; 2 for full output. (default: 0)
        'nfolds'    : The number of folds to partition data into (default: 3)
        'repetition': Determines the number of times to repeat the Feature Removal. the algorithm runs for repetitions * nfolds iterations (default: 3)
        'max_iters' : Determines the maximum number of iteration of IFR on a particular data partition fold(default: 5)
        'max_features_per_iter': Determines the maximum number of features selected for one iteration on a particular data partition fold (default: 50)
        'C'         : The value for the sparsity promoting parameter used in SSVM (Sparse SVM). (default: 1.)


        '''
        from calcom.solvers import LPPrimalDualPy

        self.params = {}

        self.params['partition_method'] = 'stratified_k-fold'   # Passed to calcom.utils.generate_partitions
        self.params['nfolds'] = 3   # Passed to calcom.utils.generate_partitions
        self.params['max_iters'] = 5    # Max iterations for IFR on one data partition
        self.params['cutoff'] = 0.75    # BSR threshold
        self.params['jumpratio'] = 100. # Relative drop needed to detect numerically zero weights in SSVM.
        self.params['repetition'] = 3   # Number of repetitions to do.
        self.params['max_features_per_iter_ratio'] = 0.8   # fraction of training data samples as cutoff for maximum features extracted per iteration 
        self.params['method'] = LPPrimalDualPy   # Default linear problem solver for SSVM
        self.params['use_cuda'] = False # flag to run SSVM on GPU
        self.params['C'] = 1.           # Sparsity promoting parameter for use in SSVM
        self.params['verbosity'] = 0    # Verbosity of print statements; make positive to see detail.
        self.params['diagnostics'] = {}
        super(IFR, self).__init__()
    #


    def _add_diagnostic_info_for_current_iteration(self, diag_dict, n_iteration, train_bsr, test_bsr,
        sorted_abs_weights, weight_ratios, features, true_feature_count, cap_breached):
        obj = {}
        obj['train_bsr'] = train_bsr
        obj['test_bsr'] = test_bsr
        obj['sorted_abs_weights'] = sorted_abs_weights
        obj['weight_ratios'] = weight_ratios
        obj['features'] = features
        obj['true_feature_count'] = true_feature_count
        obj['cap_breached'] = cap_breached
        diag_dict[n_iteration] = obj


    def _add_diagnostic_info_for_data_partition(self, diag_dict, n_data_partition, exit_reason):
        obj = {}
        obj['diagnostic_info'] = diag_dict
        obj['exit_reason'] = exit_reason
        self.params["diagnostics"][n_data_partition] = obj

    def _initialize_feature_dictionary(self, n_attributes):
        '''
        Initializes the self.feature_set with keys as all features and initial value as 0
        '''
        #initialize feature index(key) to 0 (count)
        self.feature_set = {}
        for i in range(n_attributes):
            self.feature_set[i] = 0

    def _initialize_weights_dictionary(self, n_attributes):
        '''
        Initializes the self.weights with keys as all features and empty_list
        '''
        #initialize feature index(key) to []
        self.weights = {}
        for i in range(n_attributes):
            self.weights[i] = []



    def _update_feature_dictionary(self, features):
        '''
        Increments the values of features by 1, for passed features, in self.feature_set
        '''
        #iterate over features
        for feature in features:
            #update count in feature select
            self.feature_set[feature] = self.feature_set[feature] + 1

    def _update_weights_dictionary(self, features, weights):
        '''
        Appends the weight, for passed features, in self.weights
        ''' 
        #iterate over features
        for feature,weight in zip(features, weights):
            #append the weight of the cuurent feature in self.weights
            self.weights[feature] = np.concatenate([self.weights[feature], [weight]]) 
            



    def _non_zero_features_count(self):
        '''
        returns the number of non zero features in self.feature_set
        '''
        cnt = 0
        for key in self.feature_set:
            if self.feature_set[key] > 0:
                cnt += 1
        return cnt

    def _non_zero_features(self):
        '''
        Returns the subset of self.feature_set where the values of
        self.feature_set are non zero
        '''
        non_zero_features = {}

        for feature in self.feature_set:
            if self.feature_set[feature] > 0:
                non_zero_features[feature] = self.feature_set[feature]

        return non_zero_features

    def _non_empty_weights(self):
        '''
        Returns the subset of self.weights where the list of
        self.weights are not empty
        '''
        non_empty_weights = {}

        for feature, weights in self.weights.items():
            if len(weights) > 0:
                non_empty_weights[feature] = weights

        return non_empty_weights

    def process(self, data, labels):
        '''
        data        : m-by-n array of data, with m the number of observations.
        labels      : m vector of labels for the data

        return      : a dictionary of features {keys=feature : value=no_of_time_it_was_selected}
        '''
        import calcom
        import numpy as np
        import time

        m,n = np.shape(data)

        if self.params['verbosity']>0:
            print('IFR parameters:\n')
            for k,v in self.params.items():
                print( '%20s : %s'%(k,v) )
            print('\n')
        #


        if self.params['nfolds'] < 2:
            raise ValueError("Number of folds have to be greater than 1")

        self._initialize_feature_dictionary(n)
        self._initialize_weights_dictionary(n)

        bsr = calcom.metrics.ConfusionMatrix('bsr')

        #define SSVM classifier
        ssvm = calcom.classifiers.SSVMClassifier()
        ssvm.params['C'] = self.params['C']
        ssvm.params['use_cuda'] = self.params['use_cuda']
        ssvm.params['method'] = self.params['method']
        # start processing

        #total time is the time taken by this method divided by total Number
        #of iterations it will run for.
        total_start_time = time.time()

        #method time is similar to total time, but calculates just the
        #time taken by ssvm to fit the data
        total_method_time = 0

        total_iterations = 0
        n_data_partition = 0
        for n_rep in range(self.params['repetition']):

            if self.params['verbosity']>0:
                print("=====================================================")
                print("beginning of repetition ", n_rep+1, " of ", self.params['repetition'])
                print("=====================================================")


            partitions = calcom.utils.generate_partitions(labels, method=self.params['partition_method'], nfolds=self.params['nfolds'])

            for i, partition in enumerate(partitions):

                n_data_partition +=1
                n_inner_itr = 0
                n_jump_failed = 0

                if self.params['verbosity']>0:
                    print("=====================================================")
                    print("beginning of execution for fold ", i+1, " of ", len(partitions))
                    print("=====================================================")
                #

                list_of_features_for_curr_fold = np.array([], dtype=np.int64)
                list_of_weights_for_curr_fold = np.array([], dtype=np.int64)            
                selected = np.array([], dtype=np.int64)
                # Mask array which tracks features which haven't been removed.
                active_mask = np.ones(n, dtype=bool)

                train_idx, test_idx = partition
                train_data = data[train_idx, :]
                train_labels = labels[train_idx]

                test_data = data[test_idx, :]
                test_labels = labels[test_idx]
		 #create an empty dictionary to store diagnostic info for the
                #current data partition, this dictionary has info each iteration
                #on this data partition

                diagnostic_info_dictionary = {}
                exit_reason = "max_iters"
                for i in range(self.params['max_iters']):
                    n_inner_itr += 1
                    total_iterations += 1
                    if self.params['verbosity'] > 1:
                        print("=====================================================")
                        print("beginning of inner loop iteration ", n_inner_itr)
                        print("Number of features selected for this fold: %i of %i"%(len(list_of_features_for_curr_fold),n))
                        print("Checking BSR of complementary problem... ",end="")
                    #

                    #redefine SSVM classifier with new random parameters
                    ssvm = calcom.classifiers.SSVMClassifier()
                    ssvm.params['C'] = self.params['C']
                    ssvm.params['use_cuda'] = self.params['use_cuda']
                    ssvm.params['method'] = self.params['method']

                    tr_d = np.array( train_data[:,active_mask] )
                    te_d = np.array( test_data[:,active_mask] )
                    #import pdb; pdb.set_trace()
                    try:
                        method_start_time = time.time()
                        ssvm.fit(tr_d, train_labels)
                        total_method_time += time.time() - method_start_time
                    except Exception as e:
                        if self.params['verbosity']>0:
                            print("Warning: during the training process the following exception occurred:\n")
                            print(str(e))
                            print("\nBreaking the execution for the current data fold")
                            #save the diagnostic information
                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            i,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None)
                        exit_reason = "exception_in_ssvm_fitting"
                        break
                    
                    weight = ssvm.results['weight']

                    #calculate BSR`for training data
                    pred_train = ssvm.predict(tr_d)
                    bsrval_train = bsr.evaluate(train_labels, pred_train)
                    if self.params['verbosity']>1:
                        print('')
                        print("Training BSR %.3f. "%bsrval_train)
                        print("")

                    #calculate BSR`for test data
                    pred_test = ssvm.predict(te_d)
                    bsrval_test = bsr.evaluate(test_labels, pred_test)

                    if self.params['verbosity']>1:
                        print("Testing BSR %.3f. "%bsrval_test)
                        print("")

                    #Check if BSR is above cutoff
                    if (bsrval_test < self.params['cutoff']):
                        if self.params['verbosity']>1:
                            print("BSR below cutoff, exiting inner loop.")

                        #save the diagnostic information for this iteration
                        #in this case we only have train and test bsr
                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            i,
                            bsrval_train,
                            bsrval_test,
                            None,
                            None,
                            None,
                            None,
                            None)

                        #break out of current loop if bsr is below cutoff
                        exit_reason = "test_bsr_cutoff"
                        break

                    ##########
                    #
                    # Detect where the coefficients in the weight vector are
                    # numerically zero, based on the (absolute value) ratio of
                    # successive coefficients.
                    #

                    # Look at absolute values and sort largest to smallest.
                    abs_weights = (np.abs(weight)).flatten()
                    order = np.argsort(-abs_weights)
                    sorted_abs_weights = abs_weights[order]

                    # Detect jumps in the coefficient values using a ratio parameter.
                    # jumpratios = sabsweight[:-1]/sabsweight[1:]
                    weight_ratios = sorted_abs_weights[:-1] / (sorted_abs_weights[1:] + np.finfo(float).eps)
                    jumpidxs = np.where(weight_ratios > self.params['jumpratio'])[0]


                    #check if sufficient jump was found
                    if len(jumpidxs)==0:
                        #jump never happened.
                        #save the diagnostic information for this iteration
                        #we still do not have the selected feature count and features

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            i,
                            bsrval_train,
                            bsrval_test,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "jump_failed"
                   
                   
                        #break out of the loop
                        if self.params['verbosity']>1:
                            print('There was no jump of sufficient size between ratios of successive coefficients in the weight vector.')
                            print("Discarding iteration..")
                        break

                    else:
                        count = jumpidxs[0]

                    #check if the weight at the jump is greater than cutoff
                    if sorted_abs_weights[count] < 1e-6:

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            i,
                            bsrval_train,
                            bsrval_test,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "small_weight_at_jump"
                        if self.params['verbosity']>1:
                            print('Weight at the jump(', sorted_abs_weights[count] ,')  smaller than weight cutoff(1e-6).')
                            print("Discarding iteration..")
                        break

                    count += 1
                    cap_breached = False
                    #check if the number of selected features is greater than the cap
                    if count > int(self.params['max_features_per_iter_ratio'] * train_data.shape[0]):

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            i,
                            bsrval_train,
                            bsrval_test,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "max_features_reached"
                        if self.params['verbosity']>1:
                            print('More features selected than the ', self.params['max_features_per_iter_ratio'], ' ratio of training data samples(', train_data.shape[0], ')')
                            print("Discarding iteration..")
                        
                        break

                    
                    #select features: order is list of sorted features
                    selected = order[:count]

                    if self.params['verbosity']>1:
                        print("\nSelected features on this iteration:")
                        print(selected)
                        print("\n")
                    #

                    # Selected indices are relative to the current active set.
                    # Get the mapping back to the original indices.
                    active_idxs = np.where(active_mask)[0]

                    active_mask[active_idxs[selected]] = 0

                    #append the selected features to the list_of_features_for_curr_fold
                    list_of_features_for_curr_fold = np.concatenate([list_of_features_for_curr_fold ,  active_idxs[selected]])
                    list_of_weights_for_curr_fold = np.concatenate([list_of_weights_for_curr_fold ,  weight.flatten()[order][:count]])
                    #save the diagnostic information for this iteration
                    #here we have all the information we need
                    self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                        i,
                        bsrval_train,
                        bsrval_test,
                        sorted_abs_weights,
                        weight_ratios,
                        active_idxs[selected],
                        count,
                        cap_breached)

                    if self.params['verbosity']>1:
                        print('Removing %i features from training and test matrices.'%len(selected))
                        #print("features selected in this repetition", list_of_features_for_curr_fold.shape)
                        print("\n")

                # update the feature set dictionary based on the features collected for current fold
                self._update_feature_dictionary(list_of_features_for_curr_fold)
                self._update_weights_dictionary(list_of_features_for_curr_fold, list_of_weights_for_curr_fold) 
                #save the diagnostic information for this data partition
                self._add_diagnostic_info_for_data_partition(diagnostic_info_dictionary, n_data_partition, exit_reason)

            #
        total_time_per_iteration = (time.time() - total_start_time) / total_iterations
        method_time_per_iteration = total_method_time / total_iterations


        self.params['total_time'] = total_time_per_iteration
        self.params['method_time'] = method_time_per_iteration
        if self.params['verbosity']>0:
            print("=====================================================")
            print("Finishing Execution.", self._non_zero_features_count(), " number of features were selected.")
            print("=====================================================")

        return self._non_zero_features(), self._non_empty_weights()
    #
#

if __name__ == "__main__":
    # Testing
    # from calcom import Calcom
    # cc = Calcom()
    import calcom

    # data,labels = cc.load_data('../../examples/data/CS29h.csv',shuffle=False)
    ccd = calcom.io.CCDataSet('../../geo_data_processing/ccd_gse_11study.h5')

    # A search for control data; we don't need to be too picky here.
    q0 = [['time_id',np.arange(-100,1)],['StudyID','gse73072|gse61754|gse20346|gse40012']]
    # A search for eventual influenza symptomatics anywhere between (0,5] hours post-infection.
    q1 = [
        ['StudyID', 'gse73072|gse61754|gse20346|gse40012'],
        ['disease', 'h1n1|h3n2|.*influenza'],
        ['time_id',np.arange(1,6)],
        ['symptomatic',True]
    ]

    ccd.generate_attr_from_queries('control_v_symp5hr', {'control':q0, 'symp5hr':q1}, attrdescr='A manual labeling of all controls vs post-infection influenza, eventual symptomatics up to hour five.')
    idxs = ccd.find_attr_by_value('control_v_symp5hr','control|symp5hr')

    data = ccd.generate_data_matrix(idx_list=idxs)
    labels = ccd.generate_labels('control_v_symp5hr',idx_list=idxs)

    ifr = IFR()

    ifr.params['verbosity'] = 2
    ifr.params['C'] = 1
    ifr.params['repetition'] = 5

    feature_set = ifr.process(data, labels)

    print(feature_set)
    print("Number of feature selected: %i of %i"%(len(feature_set),data.shape[1]))
#
