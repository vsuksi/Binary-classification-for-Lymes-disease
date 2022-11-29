

def createOutputAsCentroids(data,label):
	import numpy as np
	centroidLabels=np.unique(label)
	outputData=np.zeros([np.shape(data)[0],np.shape(data)[1]])
	for i in range(len(centroidLabels)):
		indices=np.where(centroidLabels[i]==label)[0]
		tmpData=data[indices,:]
		centroid=np.mean(tmpData,axis=0)
		outputData[indices,]=centroid
	return outputData

def fisherDistance(data,labels):
	'''
	labels need to be numeric
	'''
	from operator import itemgetter
	import itertools
	import numpy as np

	dataDim = np.shape(data)[1]
	noClasses = len(np.unique(labels))
	classLabels = np.unique(labels)
	lit = (((str(np.unique(labels)).replace('[','')).replace(']','')).replace(',','')).replace(' ','')
	fScore = {}
	classCombo = list(itertools.combinations(lit,2))
	labeledData = attachLabels( data,labels)
	dataStats = np.zeros([noClasses,2])
	sortedData = np.array(sorted(labeledData, key=itemgetter(dataDim)))
	for c in range(len(classLabels)):
		dataStats[c,0] = len([i for i in range(np.shape(sortedData)[0]) if sortedData[i,-1] == classLabels[c]])
	startIndex = 0
	centroids = []
	for c in range(len(classLabels)):
		if(c==0):
			tmpData = sortedData[startIndex:dataStats[c,0],:-1]
			centroids.append(np.mean(tmpData,axis=0))
			classDist = sum([ np.square(np.linalg.norm(centroids-tmpData[i])) for i in range(len(tmpData))])
			dataStats[c,1] = classDist
			startIndex = startIndex + dataStats[c,0]
		else:
			tmpData = sortedData[startIndex:startIndex+dataStats[c,0],:-1]
			centroids.append(np.mean(tmpData,axis=0))
			classDist = sum([ np.square(np.linalg.norm(centroids-tmpData[i])) for i in range(len(tmpData))])
			dataStats[c,1] = classDist
			startIndex = startIndex + dataStats[c,0]
	centroids = np.vstack((centroids))
	for combo in range(len(classCombo)):
		key = classCombo[combo][0]+classCombo[combo][1]
		centroid1 =centroids[int(key[0])]
		centroid2 =centroids[int(key[1])]
		classDist1 = dataStats[int(key[0]),1]
		classDist2 = dataStats[int(key[1]),1]
		score = np.square(np.linalg.norm(centroid1 - centroid2))/(classDist1 + classDist2)
		fScore[key] = score
	return(fScore)

def standardizeData(data,mu=[],std=[]):
	#data: a m x n matrix where m is the no of observations and n is no of features
	# try:
	# 	mu == None
	# 	b1 = True
	# except:
	# 	b1 = False
    # #
	# try:
	# 	std==None
	# 	b2 = True
	# except:
	# 	b2 = False
    # #
	import numpy as np

	if not (len(mu) and len(std)):
		std = np.std(data,axis=0)
		mu = np.mean(data,axis=0)
		constants = np.where(std==0)[0]
		std[constants] = 1
		standardizeData = (data - mu)/std
		# return mu,std,constants,standardizeData
		return mu,std,standardizeData

	else:
		standardizeData = (data - mu)/std
		# if constants != None:
		# 	standardizeData[:,constants] = 0.0 #Setting the values of the constant attributes/features to zero
		return standardizeData

def unStandardizeData(data,mu,std,constants=None):
	return std * data + mu

def fisherDistanceBinaryClass(data,labels):
#This code is written only for two classes.this can be extented for multi-classesd
#Labels can either be 0/1. [classes[0]=0 and classes[1]=1].For sepsis data 0 means normal patient and 1 mean Sepsis patient
	import numpy as np

	class1=[]
	class2=[]
	classes = np.unique(labels)
	#Now populate the lists class1 and class2
	[class1.append(data[i]) if labels[i]==classes[0] else class2.append(data[i]) for i in range(len(labels))] # using list comprehension
	class1 = np.array(class1)
	class2 = np.array(class2)
	centroid1 = np.mean(class1,0)
	centroid2 = np.mean(class2,0)
	class1Dist = sum([ np.square(np.linalg.norm(centroid1-class1[i])) for i in range(len(class1))])
	class2Dist = sum([ np.square(np.linalg.norm(centroid2-class2[i])) for i in range(len(class2))])
	return(np.square(np.linalg.norm(centroid1 - centroid2))/( class1Dist + class2Dist))

def projectRBMdata(W,train_data):
	import numpy as np

	temp_prob = np.hstack((train_data,(np.ones(np.shape(train_data)[0]))[np.newaxis].T))
	for layer in range(int(np.size(W)/2)):
		if(layer+1 == len(W)/2): # Don't apply the sigmoidal
			temp_prob = np.dot(temp_prob,W[layer])
		else:
			temp_prob = 1.0/(1 + np.exp(-np.dot(temp_prob,W[layer])))
		if(layer+1 < len(W)/2):
			temp_prob = np.hstack((temp_prob,(np.ones(np.shape(temp_prob)[0]))[np.newaxis].T))
	return temp_prob

def RegenerateRBMInput (N_W_M, test_img):
	#Module Name: Regenerate.py
	#Version: 1.0
	#Author Tomojit Ghosh
	#Purpose: This tiny little function will produce a figure which will have
	#two sub-figures. The first one is the original picture and the second one is the
	#regenerated one by RBM pretraining layer.
	#Input: Weight Matrix, an image
	#Output: A figure of two subfigures.
	import matplotlib.pyplot as plt
	import numpy as np

	temp_prob = np.append(test_img,np.array([1])[np.newaxis])
	for layer in range(int(np.size(N_W_M))):
		if(layer+1 == int(len(N_W_M)/2)): # Don't apply the sigmoidal
			temp_prob = np.dot(temp_prob,N_W_M[layer])
		else:
			temp_prob = 1.0/(1 + np.exp(-np.dot(temp_prob,N_W_M[layer])))
		if(layer < int(len(N_W_M))):
			temp_prob = np.append(temp_prob,np.array([1])[np.newaxis])

	err = abs(np.linalg.norm(test_img) - np.linalg.norm(temp_prob))
	#Now show the images
	f=plt.figure(1)
	sp1 = plt.subplot(1, 2, 1)
	plt.imshow((np.resize(test_img,(28,28))),cmap="Greys_r")
	sp1.set_title('Original Image')
	sp1.axis('off')
	sp2 = plt.subplot(1, 2, 2)
	plt.imshow((np.resize(temp_prob,(28,28))),cmap="Greys_r")
	sp2.set_title('Regenerated Image')
	sp2.axis('off')
	msg = 'Error: '+repr(err)[0:7]
	f.suptitle(msg,fontsize=16)
	plt.show()
	return temp_prob, err

def MSE(orgData,regenData):
	#orgData,regenData is n x m matrix where n is no. of points and m is no. of features.
	import numpy as np

	noSamples = np.shape(orgData)[0]
	errSum = 0
	for i in range(noSamples):
		#err = abs(np.linalg.norm(orgData[i,:]) - np.linalg.norm(regenData[i,:]))
		residual = (orgData[i,:] - regenData[i,:])
		err = np.dot(residual,residual.T)
		errSum = errSum + err
	return(errSum/noSamples)

def createIndVars(labels,bitPerClass=1):
	import numpy as np

	labels=(labels==np.unique(labels)).astype(int)
	if bitPerClass == 1:
		return labels
	else:
		return np.repeat(labels,bitPerClass,axis=1)*(1.0/bitPerClass)

def attachLabels( original_data,l):
	import numpy as np

	if(np.ndim(l) == 1):
		l=l[np.newaxis].T
	original_data = np.hstack((original_data,l))
	return original_data

def detachLabels( labeled_data):
	import numpy as np
	return labeled_data[:,:-1], labeled_data[:,-1][np.newaxis].T

def featureSelector(data,pathway_no,conf_f,class_labels):
	#This is function will take a dataset and apathway no. Then it will run ANN classification removing one
	#feature at a time and will plot the accuracy vs no of features in the dataset.This will give an idea of important
	#features in the dataset. Dataset needs to be labeled. The classifier should have a train and test method.

	import matplotlib.pyplot as plt
	import numpy as np

	iterations = 100
	split_ratio = 0.8
	no_features = np.shape(data)[1] -1#Substructing 1 to ignore the label
	h_units = int((conf_f[pathway_no-1,:3][2]).split(',')[0])
	avg_success_rate =[]
	for i in range(no_features - 1):
		labeled_data = data[:,-(no_features + 1 - i):]#Eleminate features from the left
		single_pathway_success_rate = []
		single_pathway_TPR = []
		single_pathway_TNR = []
		for l in range(iterations):
			train_data,test_data =  splitData(labeled_data,split_ratio)
			train_data,train_labels = detachLabels(train_data)
			train_labels = train_labels[np.newaxis].T
			test_data,test_labels = detachLabels(test_data)
			test_labels = test_labels[np.newaxis].T
			# print('No of features: ',np.shape(train_data)[1])
			nnet = NeuralNetworkClassifier(np.shape(train_data)[1],h_units,len(np.unique(train_labels)))
			nnet.train(train_data,train_labels,weightPrecision=0,errorPrecision=0,nIterations=100)
			#print( "scg stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason)
			(predicted_labels,Zs) = nnet.use(test_data, allOutputs=True)
			success_rate,bsr,TPR,TNR = confusionMatrix(test_labels,predicted_labels,[class_labels[0], class_labels[1]])
			single_pathway_success_rate.append(bsr)
			single_pathway_TPR.append(TPR)
			single_pathway_TNR.append(TNR)
		avg_success_rate.append(np.mean(single_pathway_success_rate))
	#Now plot the success rate vs feature index
	if(no_features > 50):
		feature_index = 50
	else:
		feature_index = no_features
	#indexList = list(reversed(np.arange(feature_index)+1))
	fig = plt.figure()
	fig_l = 'Iterative Feature Selection'
	ax = fig.add_subplot(1,1,1)
	#ax.plot(np.arange(no_fetures)+1,avg_success_rate[:no_fetures])
	# print('len(avg_success_rate): ',len(avg_success_rate))
	ax.plot(np.arange(feature_index)+1,avg_success_rate[:feature_index])
	#ax.plot(indexList,avg_success_rate[:feature_index])
	ax.grid()
	ax.set_xticks(np.arange(feature_index)+1)
	ax.set_xticklabels([no_features - j for j in range(feature_index)],rotation='vertical')
	ax.set_xlabel("No of Features")
	ax.set_ylabel("Balanced Success Rate")
	ax.set_title("Feature Selection")
	#plt.axis('tight');
	fig.suptitle(fig_l,fontsize=15)
	plt.show()
