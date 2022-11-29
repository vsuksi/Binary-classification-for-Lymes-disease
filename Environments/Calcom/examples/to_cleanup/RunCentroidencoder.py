from __future__ import absolute_import, division, print_function, unicode_literals

if __name__ == "__main__":

    from calcom.classifiers import CentroidencoderClassifier, SSVMClassifier
    from calcom.classifiers.centroidencoder.utilityDBN import standardizeData,splitData,confusionMatrix
    import numpy as np
    import pandas
    import pdb
    import calcom


    # Import the data and manually assign labels (based on prior knowledge)

    # if True:
    #     c1=np.array(pandas.read_csv('../data/PC5.csv',delimiter=',',header=None))
    #     c2=np.array(pandas.read_csv('../data/PS5.csv',delimiter=',',header=None))
    #
    #     classList = ['control5hr','shedder5hr']
    #
    #     lc1=np.zeros([len(c1),1])
    #     lc2=np.ones([len(c2),1])
    #     l=np.vstack((lc1,lc2))
    #     labeledData=np.hstack((np.vstack((c1,c2)),l))
    #     #np.savetxt('CS29.csv', labeledData, delimiter=',')
    #     #quit();
    # else:
    #     ccom = calcom.Calcom()
    #     labeledData = ccom.load_data('data/artificial.csv',labeled=False) # To fit with code below, need to use this flag to include last column.
    # #

    ssvm_l = []
    ce_l = []
    for i in range(0,20):
        # Split the data into training and test data.
        splitRatio = 0.8
        lTrainData,lTestData = splitData(labeledData,splitRatio)
        #pdb.set_trace()
        tr,trLabels,test,testLabels = lTrainData[:,:-1],lTrainData[:,-1].reshape(-1,1),lTestData[:,:-1], lTestData[:,-1].reshape(-1,1)

        # Initialize the centroid encoder classifier with some parameters.
        ce=CentroidencoderClassifier()
        hLayer = [25,3,25]
        actFuncList = ['tanh','tanh','tanh']
        errorFunc = 'MSE'
        optimizationFuncName='scg'
        noItrPre,noItrPost,noItrSoftmax=10,40,10
        noItrFinetune,batchFlag=10,'False'

        ce.initParams(hLayer,actFuncList,errorFunc,optimizationFuncName,noItrPre,noItrPost,noItrSoftmax,noItrFinetune,batchFlag)

        # Train and test.
        ce.fit(tr,trLabels)
        predictedTstLabel = ce.predict(test)

        # Evaluate results.
        confumat = calcom.metrics.ConfusionMatrix()
        confumat.evaluate(predictedTstLabel,testLabels)

        print('CE Confusion matrix:')
        print(confumat.params['cf'])
        print('CE BSR on Test Data: %3.3f'%confumat.params['bsr'])
        ce_l = ce_l + [confumat.params['bsr']]


        SSVM = SSVMClassifier();
        # Train and test.
        SSVM.fit(tr,trLabels)
        predictedTstLabel = SSVM.predict(test)

        # Evaluate results.
        confumat = calcom.metrics.ConfusionMatrix()
        confumat.evaluate(predictedTstLabel,testLabels)

        print('SSVM Confusion matrix:')
        print(confumat.params['cf'])
        print('SSVM BSR on Test Data: %3.3f'%confumat.params['bsr'])
        ssvm_l = ssvm_l + [confumat.params['bsr']]

        # print('Accuracy on traing data = ','{0:.2f}'.format(tstAccuracy))
        #print('Average balanced success rate on test data = ','{0:.2f}'.format(np.mean(tstAccuracy)),'(+/-)','{0:.2f}'.format(np.std(tstAccuracy)))
    print("CE averge:",np.average(ce_l), np.std(ce_l))
    print("SSVM average:",np.average(ssvm_l),np.std(ssvm_l))
