#from __future__ import absolute_import, division, print_function
from calcom.classifiers._abstractclassifier import AbstractClassifier

class Multiclass(AbstractClassifier):
    '''
    Purpose: This is a generic interface to convert a
    binary classifier into a multiclass classifier, based on
    one of two approaches:

    (a) a pairwise scheme;
    (b) a one-versus-all scheme (not yet implemeted).

    The pairwise scheme partitions the data by class, then
    sets up n-choose-2 binary classification models.
    Predicted class labels for test data are based on the
    majority vote across all models.

    The one-versus-all scheme partitions sets up n binary classification
    problems; one for each class. Care must be taken here, as
    this will most likely lead to unbalanced classes, which is its own
    problem. Incorporating the use of synthetic data seamlessly for this
    scheme is something we may do in the future.

    NOTE: Ties are not handled well. The function numpy.argmax()
    will return the first occurrence of the largest vote;
    e.g., a set of votes [4,4,1] will always vote for class 0.

    NOTE: This classifier is useless on its own! It must be instantiated
    with another classifier. Exceptions will be raised if a classifier
    isn't specified.
    '''
    def __init__(self,clf=None):
        self.params = {}
        self.params['classifier'] = clf
        self.params['method'] = 'pairwise'
        self.params['verbosity'] = 0

        self.params['feature_sets'] = {}
        self.params['pairwise_features'] = False
        self.results = {}
        pass
    #

    @property
    def _is_native_multiclass(self):
        return True
    #
    @property
    def _is_ensemble_method(self):
        return True

    def _fit(self,data,labels,**kwargs):
        '''
        Multiclass fit function. Given the input data and
        labels, build a collection of models from the underlying classifier.

        Inputs:
            data : numpy array size n-by-d with n datapoints in d dimensions
            labels : array or list of size n with corresponding labels.

        Optional inputs:
            feature_sets : A dictionary whose keys are the labels and values
                are lists of pointers of the corresponding subset of features
                to use for that label.

                For pairwise classification, partial information about
                feature sets is handled as follows:

                If both classes have a feature set:
                    The union of the two feature sets
                    for each of the labels is used for that classification task.
                    For example, if feature_sets = {0: [2,4,6], 1:[1,2,3,4], 2:[5,6,7,8]},
                    then the feature set used for fitting and predicting the 0-vs-1 model
                    would be [1,2,3,4,6], and for 0-vs-2, [2,4,5,6,7,8].
                If only one class has a feature set:
                    Only the one feature set is used.
                If neither class has a feature set:
                    The entire dataset is used.

                The feature sets are stored internally
                and referenced when the predict() function is used.

        Outputs:
            None.
        '''
        if not hasattr(self.params['classifier'], 'fit'):
            # What are you doing? The classifier doesn't have a fit function.
            raise AttributeError("The provided classifier \"%s\" does not have a fit() method. Have you provided a classifier to Multiclass()?"%str(self.params['classifier']))
        #


        import numpy as np
        import copy
        from calcom.io import CCList

        verb = self.params['verbosity']
        n,d = np.shape(data)

        if type(self.params['feature_sets']) == type(None):
            self.params['feature_sets'] = kwargs.get('feature_sets', {})
            self.params['feature_sets']['_default'] = CCList( np.arange(d, dtype=int) )
        else:
            # Assume that they've already specified a collection of feature sets.
            # Ignore whatever the user input.
            self.params['feature_sets']['_default'] = CCList( np.arange(d, dtype=int) )
        #

        # Map the labels to [0,...,nl-1] so we have an easy
        # time of indexing votes.
        #internal_labels = self._process_input_labels(labels)
        #internal_labels = np.array(internal_labels)
        labels = np.array(labels)

        ilmap = self._label_info['ildict']

        ul = self._label_info['unique_labels']
        ul2 = self._label_info['unique_labels_mapped']

        nl = len(ul)

        if verb>0:
            # Print a warning that a feature set hasn't been specified
            # if the user's given features for some labels but not all.
            missing_fset = []
            for l in ul:
                if l not in self.params['feature_sets']:
                    missing_fset.append(l)
            #
            if len(missing_fset)>0:
                print('Warning: the following classes do not have feature sets defined for them:')
                print(missing_fset)
                print('Default behavior described in the docstring will be used to handle these.')
        #

        # Set up equivalence classes for the... classes.
        eq = {}
        for uv in self._label_info['unique_labels_mapped']:
            # locs = np.where(uv==internal_labels)[0]
            locs = np.where(uv==labels)[0]
            eq[uv] = locs
        #

        # TODO: do we want to move the stuff inside here
        # to separate functions and set up some kind of switcher?
        if self.params['method'] == 'pairwise':
            if verb>0:
                clfname = self.params['classifier'].__class__.__name__
                print('Training multiclass %s with %s classes using pairwise scheme.'%(clfname, nl))
            #

            # Make an array to hold the pairwise trained models.
            pairwise_models = np.array(np.zeros((nl,nl)), dtype='O')

            for i,ui in enumerate(ul2):
                for uj in ul2[i+1:]:
                    if verb>1: print('Training class %s vs class %s.'%(ilmap[ui],ilmap[uj]))
                    clfcopy = copy.deepcopy(self.params['classifier'])

                    ptrs = np.union1d(eq[ui], eq[uj])

                    # Select a feature set based on the logic described in the docstring.
                    p_fset = self._select_feature_set(ui,uj, self.params['pairwise_features'])

                    datasub = data[ptrs]
                    datasub = datasub[:,p_fset]
                    # labsub = internal_labels[ptrs]
                    labsub = labels[ptrs]

                    clfcopy.fit(datasub,labsub) # Why is this overwriting self._label_info????
                    pairwise_models[ui,uj] = clfcopy
                    pairwise_models[uj,ui] = clfcopy
                #
            #
            self.results['model'] = pairwise_models

        elif self.params['method'] == 'one-vs-all':
            # TODO
            raise NotImplementedError('fit() method of multiclass technique %s not implemented yet.'%self.params['method'])
        else:
            raise NotImplementedError('fit() method of multiclass technique %s not implemented yet.'%self.params['method'])
        #

        return
    #

    def _predict(self,data):
        if not hasattr(self.params['classifier'], 'predict'):
            # What are you doing? The classifier doesn't have a fit function.
            raise AttributeError("The provided classifier \"%s\" does not have a predict() method. Have you provided a classifier to Multiclass()?"%str(self.params['classifier']))
        #

        import numpy as np
        from calcom.io import CCList

        ul2 = self._label_info['unique_labels_mapped']

        nl = len(ul2)
        n,d = np.shape(data)

        if self.params['method'] == 'pairwise':
            # Run the data through the models to get an array of predictions.
            pairwise_predictions = np.zeros( (n, nl, nl), dtype=int )
            for i,ui in enumerate(ul2):
                for uj in ul2[i+1:]:
                    # Select a feature set based on the logic described in the docstring.
                    p_fset = self._select_feature_set(ui,uj, self.params['pairwise_features'])
                    datasub = data[:,p_fset]

                    #predict the datapoints
                    pred = self.results['model'][ui,uj].predict(datasub)

                    # For each prediction, tally who won, who lost.
                    for j,p in enumerate(pred):
                        # I'm just trying to increment (i,j), where i is the
                        # predicted label in the class i vs class j model.
                        choices = [ui,uj]
                        p0 = p
                        try:
                            choices.remove(p0)
                        except:
                            if verb>1:
                                import pdb
                                pdb.set_trace()
                            else:
                                pass
                        #
                        p1 = choices[0]
                        pairwise_predictions[j,p0,p1] += 1
            #

            # With this array, the predicted class is the
            # majority vote *by row*.
            #
            # A random class is selected in the case of ties.
            # Note that this could introduce some sort of bias
            # if there's imbalanced classes.
            self.results['pairwise_predictions'] = pairwise_predictions
            votes = np.sum(pairwise_predictions, axis=2)
            max_vote_vals = np.max(votes, axis=1)
            winners = [np.where(vote==max_vote_vals[i])[0] for i,vote in enumerate(votes)]
            # pred_labels_mapped = np.argmax(votes, axis=1)
            pred_labels_mapped = [np.random.choice(w) for w in winners]
        elif self.params['method'] == 'one-vs-all':
            # TODO
            raise NotImplementedError('predict() method of multiclass technique %s not implemented yet.'%self.params['method'])
        else:
            raise NotImplementedError('predict() method of multiclass technique %s not implemented yet.'%self.params['method'])
        #

        # pred_labels = CCList([ilmap[l] for l in pred_labels_mapped])
        # self.results['pred_labels'] = pred_labels

        # pred_labels = self._process_output_labels(pred_labels_mapped)

        # return pred_labels
        return pred_labels_mapped
    #

    def visualize(self):
        pass
    #

    def _fset(self,k):
        '''
        Helper function to call the dictionary self.params['feature_sets'],
        where input values lab which are not in self.params['feature_sets'].keys()
        return self.params['feature_sets']['_default'] rather than throwing an error.

        Inputs:
            k : a label; it's expected (but not necessary) for this to be
                a key in self.params['feature_sets']
        Outputs:
            fset : self.params['feature_sets'][lab] if lab in keys, else self.params['feature_sets']['_default']
        '''
        keys = self.params['feature_sets'].keys()
        if k in keys:
            return self.params['feature_sets'][k]
        else:
            return self.params['feature_sets']['_default']
        #
    #

    def _select_feature_set(self,ui,uj, pairwise_features):
        '''
        Produces a feature set as described in the .fit() docstring.
        Copied below for posterity.

        Inputs: internal labels ui,uj
        Outputs: array of pointers; the feature set for the binary
            classification task ui vs uj.

        =====================

        If both classes have a feature set:
            The union of the two feature sets
            for each of the labels is used for that classification task.
            For example, if feature_sets = {0: [2,4,6], 1:[1,2,3,4], 2:[5,6,7,8]},
            then the feature set used for fitting and predicting the 0-vs-1 model
            would be [1,2,3,4,6], and for 0-vs-2, [2,4,5,6,7,8].
        If only one class has a feature set:
            Only the one feature set is used.
        If neither class has a feature set:
            The entire dataset is used.

        NOTE: the input labels are those used internally by
        the classifier. It's expected the user specifies the labels
        in terms of what they originally input. The information needed
        to go back and forth is generated and stored self.results['label_info']
        near the beginning of self.fit(). This function will not
        work unless self.fit() has been called.

        In the case that self.params['pairwise_features'] == True,
        the names of features must follow a strict format. For example,
        if the classes are "a", "b", and "c", then there must be
        a collection of feature sets named
            "avsb",
            "avsc",
            "bvsc"
        Note this functionality may be changed in the future
        for greater flexibility.
        '''
        from numpy import union1d

        ilmap = self._label_info['ildict']
        if pairwise_features:
            feature_label = self._get_pairwise_feature_label(ilmap[ui], ilmap[uj])
            p_fset = self._fset(feature_label)

        else:
            ui_fset_exists = (ilmap[ui] in self.params['feature_sets'])
            uj_fset_exists = (ilmap[uj] in self.params['feature_sets'])
            if ui_fset_exists and uj_fset_exists:
                p_fset = union1d( self._fset(ilmap[ui]) , self._fset(ilmap[uj]) )
            elif ui_fset_exists or uj_fset_exists:
                p_fset = self._fset(ilmap[ui]) if ui_fset_exists else self._fset(ilmap[uj])
            else:
                p_fset = self._fset('_default')
        #
        return p_fset
    #

    def _get_pairwise_feature_label(self, label1, label2):
        if label1 < label2:
            return label1 + "vs" + label2
        else:
            return label2 + "vs" + label1
#
