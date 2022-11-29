import numpy as np
import matplotlib.pyplot as plt
import random
import pandas
from operator import itemgetter
import itertools
def display2DData(data,annotation_data,fig_l):
	'''
	#This function can be used to do the scatter plot
	#annotation_data has the following fields:
	#field1: Group number starts from 1
	#field2: color code
	#field3: Shape
	#field4: Size
	#field5: Label
	'''
	fig1 = plt.figure(1)
	ax1 = fig1.add_subplot(111)
	#Now iterate for each data and plot
	group = []
	for i in range(len(annotation_data)):	   
		if(int(annotation_data[i,0]) not in group):
			ax1.scatter(data[i,0],data[i,1],s=annotation_data[i,3], edgecolors=str(annotation_data[i,1]), facecolors='None',marker=annotation_data[i,2],label = annotation_data[i,4])
			group.extend([int(annotation_data[i,0])])
		else:
			ax1.scatter(data[i,0],data[i,1],s=annotation_data[i,3], edgecolors=str(annotation_data[i,1]), facecolors='None',marker=annotation_data[i,2])
			#ax1.scatter(data[i,0],data[i,1],s=annotation_data[i,3], c=str(annotation_data[i,1]), marker=annotation_data[i,2],label = "")		   
	plt.legend(loc='upper left');
	fig1.suptitle(fig_l,fontsize=15)
	plt.show()
	
def display3DData(data,annotation_data,fig_l=''):
	'''
	#This function can be used to do the scatter plot
	#annotation_data has the following fields:
	#field1: Group number starts from 1
	#field2: color code
	#field3: Shape
	#field4: Size
	#field5: Label
	'''
	from mpl_toolkits.mplot3d import axes3d
	fig1 = plt.figure(1)
	ax1 = fig1.add_subplot(111,projection='3d')
	group = []
	for i in range(len(annotation_data)):	   
		if(int(annotation_data[i,0]) not in group):
			ax1.scatter(data[i,0],data[i,1],data[i,2],s=annotation_data[i,3].astype(int), c=str(annotation_data[i,1]), marker=annotation_data[i,2],label = annotation_data[i,4])			
			group.extend([int(annotation_data[i,0])])
		else:
			ax1.scatter(data[i,0],data[i,1],data[i,2],s=annotation_data[i,3].astype(int), c=str(annotation_data[i,1]), marker=annotation_data[i,2])
	plt.legend(loc='upper left');
	fig1.suptitle(fig_l,fontsize=15)
	ax1.set_xlabel('Dim 1',fontsize=10)
	ax1.set_ylabel('Dim 2',fontsize=10)
	ax1.set_zlabel('Dim 3',fontsize=10)
	plt.show()

	
def fisherDistance(data,labels):
	'''
	labels need to be numeric
	'''
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
	if len(mu) == 0 and len(std) == 0:
		std = np.std(data,axis=0)
		mu = np.mean(data,axis=0)
		standardizeData = (data - mu)/std
		return mu,std,standardizeData
	else:
		standardizeData = (data - mu)/std
		return standardizeData
		
def unStandardizeData(data,mu,std):
	return std * data + mu

def fisherDistanceBinaryClass(data,labels):
#This code is written only for two classes.this can be extented for multi-classesd
#Labels can either be 0/1. [classes[0]=0 and classes[1]=1].For sepsis data 0 means normal patient and 1 mean Sepsis patient
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
	noSamples = np.shape(orgData)[0]	
	errSum = 0
	for i in range(noSamples):
		#err = abs(np.linalg.norm(orgData[i,:]) - np.linalg.norm(regenData[i,:]))
		residual = (orgData[i,:] - regenData[i,:])
		err = np.dot(residual,residual.T)
		errSum = errSum + err	
	return(errSum/noSamples)
	
def createIndVars(labels,bitPerClass=1):
	labels=(labels==np.unique(labels)).astype(int)
	if bitPerClass == 1:
		return labels
	else:
		return np.repeat(labels,bitPerClass,axis=1)*(1.0/bitPerClass)

def Regenerate (orgImg,testImg):
	f=plt.figure(1)
	sp1 = plt.subplot(1, 2, 1)
	plt.imshow((np.resize(orgImg,(28,28))),cmap="Greys_r")
	sp1.set_title('Original Image')
	sp1.axis('off')
	sp2 = plt.subplot(1, 2, 2)
	plt.imshow((np.resize(testImg,(28,28))),cmap="Greys_r")
	sp2.set_title('Regenerated Image')
	sp2.axis('off')
	plt.show()
	
def splitMNISTData(fName,fLabelName,no):
	#Purpose: This function is particularly written for MNIST data.
	#This function will return a data set which is a subset of the original data
	#read from the file 'fName'. For MNIST test data set I want to create a small	
	#set only comprising 50 data points for all the class[digit0 .. digit9]
	#If 'no' is 0 the I'll split the data based on the ration.
	data = np.array(pandas.read_csv(fName,header=None,delimiter=','))
	#pdb.set_trace()
	labels = np.array(pandas.read_csv(fLabelName,header=None,delimiter=','))
	noLabels = np.shape(labels)[0]
	classDataCount = np.array([sum(labels)])
	newData = []
	startIndex = 0
	newLabels = []
	for i in range(np.shape(classDataCount)[1]):
		if(i==0):
			l = np.arange(startIndex,classDataCount[0,i],1)
			random.shuffle(l)
			#print(np.shape(l))
			l = list(l)
			newData.append(data[l[:no],:])
			startIndex = startIndex + classDataCount[0,i]
		else:
			l = np.arange(startIndex,startIndex+classDataCount[0,i],1)
			random.shuffle(l)
			l = list(l)
			newData.append(data[l[:no],:])
			startIndex = startIndex + classDataCount[0,i]
		newLabels.append((np.ones(no)*i)[np.newaxis].T)
		
	return np.vstack((newData)),(np.vstack((newLabels))).astype(int)

def splitMNISTDataByClass(fName,fLabelName):
	#Purpose: To split the MNIST data by class(10, digit0 ... digit9).
	#It'll return an dictionary with data and indicator variable.
	MNISTData={}
	data=np.array(pandas.read_csv(fName,header=None,delimiter=','))
	labels=np.array(pandas.read_csv(fLabelName,header=None,delimiter=','))
	counter=0
	nData=len(data)
	indVars=(labels==np.unique(labels)).astype(int)
	for i in range(10):#As there are 10 classes
		zeroL=np.repeat(np.zeros(10).reshape(1,-1),nData,0)
		key='Digit'+str(i)
		MNISTData[key]={'data':[],'indVar':[],'indVarB':[],'label':[]}
		MNISTData[key]['data']=data[np.where(labels==i)[0],:]
		zeroL[counter:counter+len(labels[labels==i]),:]=indVars[counter:counter+len(labels[labels==i]),:].astype(int)
		#MNISTData[key]['indVar']=zeroL
		MNISTData[key]['indVar']=zeroL[np.where(labels==i)[0],:]
		MNISTData[key]['label']=labels[np.where(labels==i)[0],:]
		counter+=len(labels[labels==i])
		L=np.zeros(nData).reshape(-1,1)
		indexSet=[j for j in range(10) if j!=i]
		for j in range(len(indexSet)):
			L=L+indVars[:,indexSet[j]].reshape(-1,1)
		MNISTData[key]['indVarB']=np.hstack((indVars[:,i].reshape(-1,1),L)).astype(int)
	MNISTData['data']=data
	MNISTData['label']=labels
	MNISTData['indVar']=indVars
	return MNISTData

def findRepClassMember(data,labes):
	#Purpose: To find a good representative for each class of MNIST data
	#The representative should be farthest from the other class
	cLabels=np.unique(labels)
	for c in cLabels:
		indexSet=np.where(labels==c)[0]
		classData=data[indexSet,:]
		#for d in classData:
			

def Ebola_PCA(pca_data,m):
	org_data = pca_data
	pca_data = pca_data - np.mean(pca_data,axis=0)
	cov = np.dot(pca_data.T,pca_data)
	#pdb.set_trace()
	Eigen_val,Eigen_vec = np.linalg.eig(cov)
	#Now project the data on the first m eigen vector
	projected_data = np.dot(org_data,Eigen_vec[:,0:m])
	return projected_data,Eigen_vec[:m]
	
def attachLabels( original_data,l):
	if(np.ndim(l) == 1):
		l=l[np.newaxis].T
	original_data = np.hstack((original_data,l))
	return original_data
		
def detachLabels( labeled_data):
	return labeled_data[:,:-1], labeled_data[:,-1][np.newaxis].T
	
def splitData(labeled_data,split_ratio):
	#This method will split the dataset into training and test set based on the split ratio.
	#Training and test set will have data from each class according to the split ratio.
	#First hold the data of different classes in different variables.
	train_set =[]
	test_set =[]
	no_data = len(labeled_data)	
	sorted_data = labeled_data[np.argsort(labeled_data[:,-1])]#sorting based on the numeric label.
	first_time = 'Y'
	for classes in np.unique(sorted_data[:,-1]):
		temp_class = np.array([sorted_data[i] for i in range(no_data) if sorted_data[i,-1] == classes])
		np.random.shuffle(temp_class)#Shuffle the data so that we'll get variation in each run
		tr_samples = np.floor(len(temp_class)*split_ratio)
		tst_samples = len(temp_class) - tr_samples
		if(first_time == 'Y'):
			train_set = temp_class[:int(tr_samples),]
			test_set = temp_class[-int(tst_samples):,]
			first_time = 'N'
		else:
			train_set = np.vstack((train_set,temp_class[:int(tr_samples),]))
			test_set = np.vstack((test_set,temp_class[-int(tst_samples):,]))
	
	#no_of_trn_samples = int(np.ceil(split_ratio*len(labeled_data)))
	#no_of_tst_samples = int(len(labeled_data) - no_of_trn_samples)
	#return labeled_data[:no_of_trn_samples,:],labeled_data[-no_of_tst_samples:,:]
	return train_set,test_set
	
def confusionMatrix(actual_label_list,predicted_label_list,classLabelName,display_flag='N'):
	#This method is only applicable for binary class
	#Assumption: labels have to be numeric, example 0/1, 1/2 etc.
	#classLabelName is a list of string. 1st element of the list correspond to first numeric class label (after ascending sort)
	#For binary classification the confusion matrix will have 4 categories of data:
	#AL0_PL0(actula label 0 and predicted label 0); AL0_PL1(actula label 0 and predicted label 1); AL1_PL0(actula label 1 and predicted label 0); AL1_PL1(actula label 1 and predicted label 1)
	#I'll create a dictionay for store all these values.
	
	confMatx = {'AL0_PL0': 0, 'AL0_PL1': 0, 'AL1_PL0': 0, 'AL1_PL1': 0}
	classLabels = np.unique(actual_label_list)
	#pdb.set_trace()
	for key in confMatx.keys():
		al=int(key[2])
		pl=int(key[6])
		confMatx[key] = float(len([j for j in range(int(len(actual_label_list))) if((actual_label_list[j,0] == al) and (predicted_label_list[j,0] == pl))]))
	#print('confMatx: ')
	#print(confMatx)
	if(display_flag == 'Y'):
		print ("Confusion Matrix:")
		print()
		print('\t', classLabelName[0],'\t', classLabelName[1])
	    
	no_of_correct_prediction = 0	
	for i in range(int(len(classLabelName))):		
		temp = []
		for j in range(int(len(classLabelName))):			
			key = 'AL'+str(i)+'_'+'PL'+str(j)
			if(i == j):
				no_of_correct_prediction = no_of_correct_prediction + confMatx[key]
			temp.append(str(confMatx[key]))	
		if(display_flag == 'Y'):
			print("{0}    {1}              {2}".format(str(classLabelName[i]),temp[0],temp[1]))
	succes_rate = 100*(no_of_correct_prediction/len(actual_label_list))	
	#succes_rate = 100*((confMatx['AL0_PL0'] + confMatx['AL1_PL1'])/len(actual_label_list))		
	if(len(classLabels) == 1):	
		bsr = 0.5*succes_rate		
	else:		
		bsr = 100*(0.5*(confMatx['AL0_PL0']/(confMatx['AL0_PL0'] + confMatx['AL0_PL1'])) + 0.5*(confMatx['AL1_PL1']/(confMatx['AL1_PL1'] + confMatx['AL1_PL0'])))
	if(display_flag == 'Y'):
		print()
		print('Success Rate: %.2f'%succes_rate)
		print ('Balanced Success Rate: %.2f'%bsr)
	#Important comments: In this code label 1 means susceptible/positive class(Ebola) and label 0 means resistant/negative(not Ebola).
	#AL1_PL1:actuallly Ebola and also predicted as Ebola.TP(true positive).
	#AL0_PL1:actuallly not Ebola but predicted as Ebola.FP(false positive).
	#AL1_PL0:actuallly Ebola but predicted as not Ebola(resistant).FN(false negitive).
	#AL0_PL0:actuallly not Ebola and also predicted as not Ebola.TN(true negitive).
	TP,FP,FN,TN = confMatx['AL1_PL1'],confMatx['AL0_PL1'],confMatx['AL1_PL0'],confMatx['AL0_PL0']
	if(TP == 0):
		TPR = 0
	else:
		TPR = 100*(TP/(TP+FN))
	if(TN == 0):
		TNR = 0
	else:
		TNR = 100*(TN/(FP+TN))
	return succes_rate,bsr,TPR,TNR

def featureSelector(data,pathway_no,conf_f,class_labels):
	#This is function will take a dataset and apathway no. Then it will run ANN classification removing one 
	#feature at a time and will plot the accuracy vs no of features in the dataset.This will give an idea of important
	#features in the dataset. Dataset needs to be labeled. The classifier should have a train and test method.
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
			print('No of features: ',np.shape(train_data)[1])
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
	print('len(avg_success_rate): ',len(avg_success_rate))	
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
