# -*- coding: utf-8 -*-

"""
    Console script for calcom.
    *** Important: This class is deprecated. Needs to be updated/rewritten to utilize new functionalities. ***
    
"""

from __future__ import absolute_import, division, print_function


import sklearn
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import *


import click
import calcom
from .calcom import Calcom
from .experiment import Experiment
import calcom.classifiers, calcom.metrics
import os

@click.group()
def main():
    pass

@main.group()
def create():
    pass

@create.command()
@click.option('--inputfile',prompt='Path to input file', help='Path to the input file.')
@click.option('--classifier_list', prompt='Classifiers(List as comma seperated values)', help='List of classifiers on which the experiment is going to run on.')
@click.option('--custom_classifier_list', default='', help='List of custom classifiers on which the experiment is going to run on.')
@click.option('--metric', default='bsr', type=click.Choice(['tpr','tnr','ppv','npv', 'fnr', 'fpr', 'fdr', 'for', 'acc', 'bsr']), help='Specify a metric that will determine the best version of each classifiers in the classifier list. \
                                                    OPTIONS: \
                                                    tpr: True positive rate \
                                                    tnr: True negative rate \
                                                    ppv: Positive predictive value/precision \
                                                    npv: Negative predicitve value \
                                                    fnr: False negative rate/miss rate \
                                                    fpr: False positive rate/fall-out \
                                                    fdr: False discovery rate \
                                                    for: False omission rate \
                                                    acc: Accuracy \
                                                    bsr: Balanced success rate')
@click.option('--cross_validation', default='stratified_k-fold', help='Specify the type of cross-validation. (default: stratified_k-fold)\
                                                                                            OPTIONS: \
                                                                                            stratified_k-fold \
                                                                                            k-fold\
                                                                                            split-<num>:<num>, i.e. "split-80:20"')
@click.option('--nfolds', default=5, help='Specify the number of folds for cross-validation.')
@click.option('--visualize', default='n', type=click.Choice(['y','n']), help='Specify whether to visualize the classifier.[default:y]')
@click.option('--num_iter', default=1, help='Specify the number of time the experiment should run. Mean result will be output.', type=int)
@click.option('--verbosity', default=1, help='Specify the level of verbose.', type=int)
@click.option('--label_index', default=-1, help='Specify the column index of the labels')
@click.option('--balance_data', default='', help='Specify the balancing technique \
                                                    OPTIONS:\
                                                    smote:<k>, where k is the number of nearest neighbours. i.e. "smote:5".\
                                                    If, `k` is unspecified, k=5 by default.')
def experiment(inputfile,classifier_list,custom_classifier_list,metric,cross_validation,nfolds,visualize,num_iter,verbosity,label_index,balance_data):
    '''
    Run an expriment on a given list of classifiers

    Example:
        calcom create experiment --inputfile='examples/CS5h.csv' \ 
                                 --classifier_list=NMFClassifier,SSVMClassifier,CentroidencoderClassifier,GrModel \
                                 --custom_classifier_list= \
                                 --metric=bsr \
                                 --cross_validation=stratified_k-fold \
                                 --nfolds=5 \
                                 --visualize=y \
                                 --num_iter=1 \
                                 --verbosity=1
    '''
    
    # create a Calcom object and load the input data
    ccom = Calcom()
    data,labels = ccom.load_data(inputfile,label_index=label_index)



    # create list of classifier objects
    classifier_string_arr = classifier_list.split(',')

    classifier_object_arr = []
    if len(classifier_list) > 0:
        # import pickle
        for classifier_string in classifier_string_arr:
            if classifier_string.startswith("sklearn"):  #Handle sklearn classifiers
                sk_arr = classifier_string.split('.')
                #get the classifier class into c1
                c1=sklearn
                for s in sk_arr[1:]:
                    c1 = getattr(c1,s)
                #try to read params file
                filepath = os.path.expanduser("~/calcom/classifiers/params/sklearn/" + classifier_string + ".params")
                params = {}
                if os.path.exists(filepath):
                    # with open(filepath,"rb") as f:
                    #     params = pickle.load(f)
                    params = calcom.io.load_pkl(filepath)

                # create classifier object
                clf = c1(**params)
                print(clf)
                classifier_object_arr = classifier_object_arr + [clf]
            else:
                classifier_object_arr = classifier_object_arr + [load_class(classifier_string,type='calcom.classifiers')()]


    custom_classifier_string_arr = custom_classifier_list.split(',')

    if len(custom_classifier_list) > 0:
        for classifier_string in custom_classifier_string_arr:
            classifier_object_arr = classifier_object_arr + [load_class_from_file(module=classifier_string,classname=calcom.classifiers.AbstractClassifier,classname_str='AbstractClassifier',path='.')()]

    # create metric object
    metric_object = calcom.metrics.ConfusionMatrix(metric);


    import calcom.plot_wrapper as plotter
    from matplotlib import pyplot
    # Run experiment
    mean_results = {}
    for iteration in range(num_iter):
        if (verbosity>=1):
            print("Iteration: ", iteration+1)
        exp = Experiment(data = data,
                         labels = labels,
                         classifier_list = classifier_object_arr,#[load_module_util('RandomClassifier')(), classifiers.NMFClassifier()] ,
                         cross_validation = cross_validation,
                         evaluation_metric = metric_object,
                         folds=nfolds,
                         verbosity=verbosity,
                         balance_data=balance_data)
        best_classification_models = exp.run()

        # calculate the means
        for key in exp.classifier_results.keys():
            if iteration == 0:
                mean_results[key] = {
                    'max' : exp.classifier_results[key]['max'],
                    'min' : exp.classifier_results[key]['min'],
                    'mean': exp.classifier_results[key]['mean'],
                    'std' : exp.classifier_results[key]['std']
                }
            else:
                mean_results[key] = {
                    'max' : (exp.classifier_results[key]['max'] + iteration*mean_results[key]['max'])/(iteration+1),
                    'min' : (exp.classifier_results[key]['min'] + iteration*mean_results[key]['min'])/(iteration+1),
                    'mean': (exp.classifier_results[key]['mean'] + iteration*mean_results[key]['mean'])/(iteration+1),
                    'std' : (exp.classifier_results[key]['std'] + iteration*mean_results[key]['std'])/(iteration+1)
                }
        # visualize of flag enabled
        if visualize == 'y':
            print("Plotting\n")
            fig = plotter.scatterExperiment(exp, data, labels, readable_label_map={0:'Control',1:'Shedder'})
            pyplot.show(block=False)


    # print output
    print("\n")
    print("%s" % "-"*50)
    print("\n")
    print("Mean Results.\n")

    for key in exp.classifier_results.keys():
        print(metric + " statistics for %s:\n"%key)
        print( ("%-10s : %.3f \xB1 %.3f") % ("Mean",mean_results[key]['mean'],mean_results[key]['std']) )
        print( "%-10s : %.3f" % ("Maximum",mean_results[key]['max']) )
        print( "%-10s : %.3f" % ("Minimum",mean_results[key]['min']) )

        print("%s" % ("-"*20))
        print('\n')

    input("Press Enter to continue...")



@main.command()
@click.option('--inputfile',prompt='Path to input data file', help='Path to the input data file.')
@click.option('--classifier', prompt='Classifier', help='Classifier for the hyperparameter search.')
@click.option('--search_type', default='GridSearch', help='Type of search to do over the defined search space.')
@click.option('--param_grid', prompt='Specify the search space(param_grid):', help="Specify the search space \
                                                    Example:\
                                                    \"{'C': [1, 10, 100, 1000], 'kernel': ['linear']}\"")
@click.option('--metric', default='bsr', type=click.Choice(['tpr','tnr','ppv','npv', 'fnr', 'fpr', 'fdr', 'for', 'acc', 'bsr']), help='Specify a metric that will determine the best version of each classifiers in the classifier list. \
                                                    OPTIONS: \
                                                    tpr: True positive rate \
                                                    tnr: True negative rate \
                                                    ppv: Positive predictive value/precision \
                                                    npv: Negative predicitve value \
                                                    fnr: False negative rate/miss rate \
                                                    fpr: False positive rate/fall-out \
                                                    fdr: False discovery rate \
                                                    for: False omission rate \
                                                    acc: Accuracy \
                                                    bsr: Balanced success rate')
@click.option('--cross_validation', default='stratified_k-fold', help='Specify the type of cross-validation. (default: stratified_k-fold)\
                                                                                            OPTIONS: \
                                                                                            stratified_k-fold \
                                                                                            k-fold\
                                                                                            split-<num>:<num>, i.e. "split-80:20"')
@click.option('--nfolds', default=5, help='Specify the number of folds for cross-validation.')
@click.option('--verbosity', default=1, help='Specify the level of verbose.', type=int)
@click.option('--balance_data', default='', help='Specify the balancing technique \
                                                    OPTIONS:\
                                                    smote:<k>, where k is the number of nearest neighbours. i.e. "smote:5".\
                                                    If, `k` is unspecified, k=5 by default.')

@click.option('--label_index', default=-1, help='Specify the column index of the labels in the input file.')

def parameter_search(inputfile,classifier,search_type,metric,cross_validation,nfolds,verbosity,balance_data, param_grid, label_index):
    '''
    Command line interface for doing parameter search. Supported search types
    are GridSearch, BayesianOptimization

    Example:
        ``calcom parameter_search --inputfile=data/CS5h.csv --classifier=SSVMClassifier --search_type=GridSearch --param_grid="{'C':[1,2,3,4,10],'kernel': ['linear','rbf']}"``
    '''

    ccom = Calcom()

    data,labels = ccom.load_data(inputfile,label_index=label_index)

    metricObj = calcom.metrics.ConfusionMatrix(metric)
    
    if classifier.startswith("sklearn"):  #Handle sklearn classifiers
        sk_arr = classifier.split('.')
        #get the classifier class into c1
        c1=sklearn
        for s in sk_arr[1:]:
            c1 = getattr(c1,s)
        #try to read params file
        filepath = os.path.expanduser("~/calcom/classifiers/params/sklearn/" + classifier + ".params")
        params = {}
        if os.path.exists(filepath):
            params = calcom.io.load_pkl(filepath)

        # create classifier object
        clf = c1(**params)
    else:    
        try:
            clf = load_class(classifier,type='calcom.classifiers')
        except:
            raise ValueError(classifier + " : classifier name not found.")

    import ast
    from calcom import GridSearch
    gridSearch = GridSearch(
                     classifier = clf ,
                     param_grid = ast.literal_eval(param_grid),
                     cross_validation = cross_validation,
                     evaluation_metric = metricObj,
                     balance_data=balance_data,
                     verbosity=verbosity,
                     folds=nfolds)

    best_combination = gridSearch.run(data,labels)
    print("Best set of params: ", best_combination)




@main.command()
@click.argument('classifier')
@click.option('--reset', is_flag=True, help='Reset the saved parameters.')
def set_params(classifier,reset):
    '''
    Set default parameters for a classifier.

    Example:
        ``calcom set_params SSVMClassifier``
        
        Then when prompted enter the new default values. Just press
        <Return/Enter> in the keyboard if you don't want to change any of the
        parameters. Next time the classifier is run using calcom the new
        default parameters will be used.
        
        .. note:
            Please use quotes only when entering strings as parameter values.

        use01Labels(False) :
        inputDim(0) :
        C(1.0) :2
        errorTrace(None) :
        TOL(0.001) :
    '''

    dump_dict = {}
    if(reset): #if --reset flag is set remove the .params file if exists; then return
        filepath = os.path.expanduser("~/calcom/classifiers/params/sklearn/" + classifier + ".params")
        if os.path.exists(filepath):
            os.remove(filepath)
            print("Saved parameters for the classifier "+classifier+" have been reset.")
        else:
            print("No saved parameters for the classifier "+classifier)
        return
    #
    print("Note: Please use qoutes only when entering strings as paremeter values.")
    if classifier.startswith("sklearn"):
        filepath = os.path.expanduser("~/calcom/classifiers/params/sklearn/" + classifier + ".params")
        f_params = {} #load params from file if exists
        if os.path.exists(filepath):
            # with open(filepath,"rb") as f:
            #     import pickle
            #     f_params = pickle.load(f)
            f_params = calcom.io.load_pkl(filepath)

        sk_arr = classifier.split('.')
        clf=sklearn
        for s in sk_arr[1:]:
            clf = getattr(clf,s)
        import inspect
        for v in inspect.signature(clf.__init__).parameters.values(): #iterate over the parameters of __init__
            a = str(v).split('=')
            if a[0] in ['kwargs', 'args', 'self']: #ignore these parameters
                continue
            if len(a) > 1:   #if default value is set
                key,current = a
                if key in f_params: #override by the values saved in .params file
                    current = f_params[key]
                val = input(str(key)+"("+str(current)+") :")
            else:  #if default value not set
                key = a
                val = input(str(key)+":")
            if len(val) > 0:
                dump_dict[key] = eval(val) #ast.literal_eval is more secure; but the flexibility of using objects as parameters is necessary here
        path = os.path.expanduser("~/calcom/classifiers/params/sklearn/")
        
        #try to instantiate the classifier and make sure that the format of the inputs are not wrong; otherwise don't save and raise ValueError
        try:
            clf(**dump_dict)
        except:
            raise ValueError("Classifier does not accept the given parameters.")
        
    else:
        # load the classifer
        clf = load_class(classifier,type='calcom.classifiers')()
        import ast
        for key in clf.params:
            current = clf.params[key]
            if isinstance(current, str):
                current = "'" + current + "'"
            val = input(str(key)+"("+str(current)+") :")
            if len(val) > 0:
                clf.params[key] = ast.literal_eval(val)
        #
        dump_dict = clf.params

        # save new parameters into a file in the home directory
        path = os.path.expanduser("~/calcom/classifiers/params/")
        
        #try to instantiate the classifier and make sure that the format of the inputs are not wrong; otherwise don't save and raise ValueError
        try:
            clf.init_params(dump_dict)
        except:
            raise ValueError("Classifier "+classifier+" does not accept the given parameters.")




    if os.path.isdir(path) == False:
        os.makedirs(path)

    filepath = path +classifier + ".params"
    # import pickle
    # with open(filepath,"wb") as f:
    #     pickle.dump(dump_dict,f)
    calcom.io.save_pkl(dump_dict,filepath)

    # with open(filepath,"rb") as f:
    #     o = pickle.load(f)
    #     print(o)
    o = calcom.io.load_pkl(filepath)
    print(o)

@main.command()
@click.option('--visualizer',prompt='Visualizer', help='Specify the visualizer.')
@click.option('--inputfile',prompt='Path to input file', help='Relative path to the input file.')
@click.option('--label_map', default='', help='Map the labels in the dataset into human readable names (i.e. "{0.0:\'Control\',1.0:\'Shedder\'}" including the double quotes)\nAlternatively, specify a filepath containg the mapping')
@click.option('--is_labeled', default='y', type=click.Choice(['y','n']), help='Is the input data labeled?[default:y]')
@click.option('--use_defaults',default='y', type=click.Choice(['y','n']),help='Whether to use default parameters.[default:y]')
@click.option('--label_index', default=-1, help='Specify the column index of the labels.[default:-1]')
@click.option('--dim', default=3, type=int, help='Specify the dimensionality of projection. Either 2 or 3.[default:3]')
def visualize(visualizer, inputfile, label_map, is_labeled, use_defaults,label_index,dim):
    '''
    Run a visualizer.
    '''
    ccom = Calcom()
    viz = load_class(visualizer,type='calcom.visualizers')()
    if dim < 2 or dim > 3:
        raise ValueError("Invalid dimension specified. Specify either 2 or 3.")
    # if use_defaults flag is not set then get params for the visualizer and set them
    if use_defaults != 'y' and hasattr(viz, 'params'):
        import ast
        for key in viz.params:
            #print(key,"(",clf.params[key],"):")
            current = viz.params[key]
            if isinstance(current, str):
                current = "'" + current + "'"
            val = input(str(key)+"("+str(current)+") :")
            if len(val) > 0:
                viz.params[key] = ast.literal_eval(val)

    if is_labeled == 'y':
        data,labels = ccom.load_data(inputfile, labeled=True, label_index=label_index)
        if len(label_map) > 0:
            import os.path
            if os.path.isfile(label_map):
                path = os.path.expanduser(label_map)
                with open(path) as f:
                    label_map = f.read()
            import ast
            label_map = ast.literal_eval(label_map)
            coords = viz.project(data,labels,label_map,dim=dim)
        else:
            coords = viz.project(data,labels,dim=dim)

    else:
        data = ccom.load_data(inputfile, labeled=False)
        coords = viz.project(data,dim=dim)
    viz.visualize(coords)

# Utility classes
import importlib
def load_class(classname,type='calcom.classifiers'):
    '''
    Given the classname as string return the class
    '''
    l = classname
    X = importlib.import_module(type)
    C = getattr(X,classname)
    return C


def load_class_from_file(module,classname,classname_str,path=None):
    '''
    Given the classname and the path (the class does not have to be part of
    calcom) as string return the class
    '''
    if path:
        import sys
        sys.path.append(os.path.expanduser(path))
    X = importlib.import_module(module, package=None)
    C = None
    for i in dir(X):
        C = getattr(X,i)
        import inspect
        if i != classname_str and inspect.isclass(C) and issubclass(C,classname):
            break
    else:
        raise NotImplementedError("No class in the module "+ module +" extends " + classname_str)
    return C



#if __name__ == "__main__":
#main()
