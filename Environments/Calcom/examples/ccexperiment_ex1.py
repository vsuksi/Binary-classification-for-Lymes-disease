
if __name__ == "__main__":
    import calcom
    from calcom.classifiers import RandomClassifier,SSVMClassifier
    from sklearn import svm, ensemble

    import numpy as np

    # Load data
    ccd = calcom.io.CCDataSet('../../geo_data_processing/ccd_gse40012.h5')

    # Generate a new attribute; are you at 24 hours or not?
    tvals = ccd.get_attr_values('time_id')
    firstday = (tvals==24)
    ccd.append_attr('firstday',firstday)

    # Set up a cross-validation scheme which does not
    # mix a single subject's data between the training and
    # testing sets, and classifies based on the new attribute we defined.
    #
    # Run over three different classifiers to compare results.
    bsr = calcom.metrics.ConfusionMatrix('bsr')
    ccexp = calcom.CCExperiment(
        classifier_list=[
            RandomClassifier(),
            ensemble.RandomForestClassifier(),
            SSVMClassifier()
            ],
        ccd=ccd,
        evaluation_metric = bsr,
        classification_attr="firstday",
        cross_validation_attr="SubjectID",
        folds = 5,
        cross_validation='stratified_k-fold',
        verbosity = 1
    )

    best_classifiers = ccexp.run()
    # print(best_classifiers)

    # p= best_classifiers['RandomForestClassifier'].predict(ccd.generate_data_matrix())
    #
    # measure = bsr.evaluate(ccd.generate_labels('time_id'),p)
    # print(measure)
