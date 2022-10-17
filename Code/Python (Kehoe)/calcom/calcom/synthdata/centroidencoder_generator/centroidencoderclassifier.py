import pdb
import numpy as np
from copy import copy
import Autoencoder as ae
import deepANNClassifier as dc
import scaledconjugategradient as scg
import numpy as np
from utilityDBN import standardizeData
from sklearn.cluster import KMeans

class CentroidencoderClassifier():
	def __init__(self):
		'''
		Setup default parameters
		'''
		self.parameters = {}
		self.testClassifier = None

	def initParams(self,hLayer,actFunc,errorFunc,optimizationFuncName,noItrPre,noItrPost,noItrSoftmax,noItrFinetune,batchFlag,repeatCE,useSyntheticData):
		self.params = {}
		self.params['hLayer'] = hLayer
		self.params['actFunc'] = actFunc
		self.params['errorFunc'] = errorFunc
		self.params['optimizationFuncName'] = optimizationFuncName
		self.params['noItrPre'] = noItrPre
		self.params['noItrPost'] = noItrPost
		self.params['noItrSoftmax'] = noItrSoftmax
		self.params['noItrFinetune'] = noItrFinetune
		self.params['batchFlag'] = batchFlag
		self.params['repeatCE'] = repeatCE #Param to enlarge training dataset by adding ce output with the original training data.
		self.params['useSyntheticData'] = useSyntheticData

	def feval(self,fName,*args):
		return eval(fName)(*args)

	def tanh(self,X):
		return np.tanh(X)

	def sigmoid(self,X):

		return expit(X) #using the sigmoidal of scipy package

	def softplus(self,X):
		return np.log(1+np.exp(X))

	def rect(self,X):
		X[X<=0]=0*X[X<=0]
		return X

	def rectl(self,X):
		X[X<=0]=0.01*X[X<=0]
		return X

	def linear(self,X):
		return X

	def calcLogProb(self,cOut):
		'''
		target=np.exp(cOut)
		denom=np.repeat(np.sum(target,1).reshape(-1,1),self.outputDim,axis=1)
		return target/denom
		'''

		#Trick to avoid Overflow using Chuck's code
		#pdb.set_trace()
		mx = np.max(cOut)
		cOut=np.exp(cOut-mx)
		denom=np.exp(-mx) + np.sum(cOut,axis=1).reshape((-1,1))
		rowsHavingZeroDenom = denom == 0.0
		if np.sum(rowsHavingZeroDenom) > 0:
			Yshape = (cOut.shape[0],cOut.shape[1])
			nClasses = Yshape[1]
			Y = np.ones(Yshape) * 1.0/nClasses + np.random.uniform(0,0.1,Yshape)
			Y /= np.sum(Y,1).reshape((-1,1))
		else:
			Y=cOut/denom
		return Y

	def createVoronoiRegion(self,data,labels,param):
		if param['algo'] == 'kMean':
			return(self.KMean(data,labels,param))
		elif param['algo'] == 'compLrn':
			return(self.compLearning(self,data,param))
		else:
			return(self.agglomerativeClustering(self,data,param))

	def KMean(self,data,labels,param):
		nCenter = param['noCenter']
		noItr = param['nItr']
		print('Running kMeans clustering with no of centers:',nCenter)
		kmeans = KMeans(n_clusters=nCenter, n_init=50, max_iter=noItr, random_state=0).fit(data)
		centers = kmeans.cluster_centers_
		#pdb.set_trace()
		vRegion = {}
		for c in range(nCenter):
			key = 'Region'+str(c+1)
			vRegion[key] = {}
			vRegion[key]['Center'] = kmeans.cluster_centers_[c]
			vRegion[key]['PatternIndex'] = np.where(c == kmeans.labels_)[0]
		vRegion['Inertia']=np.sqrt(kmeans.inertia_)/nCenter #Inertia=Sum of the squared distances of samples from their nearest cluster center.
		print('Inertia per cluster:',vRegion['Inertia'])
		return vRegion

	def createCEData(self,trData,trLabels,param):
		if param['noCenter']==1:
			#Now create input and output for centroidencoder. Output is a representative for each class. I'm taking the centroid.
			trInput=[]
			trOutput=[]
			valInput=[]
			valOutput=[]
			trClassVal=[]
			valClassVal=[]
			for c in np.unique(trLabels):
				tmpD = trData[np.where(trLabels==c)[0],:]
				#print('No of samples in class ',c,' is ',len(tmpD))
				noP=np.shape(tmpD)[0]
				trOutput.append(np.tile(np.mean(tmpD,axis=0),(noP,1)))
				trInput.append(tmpD)
			trInput=np.vstack((trInput))
			trOutput=np.vstack((trOutput))
		else:
			trInput=[]
			trOutput=[]
			vRegion=self.createVoronoiRegion(trData,trLabels,param)
			for k in vRegion.keys():
				if 'Region' in k:
					d=trData[vRegion[k]['PatternIndex'],:]
					noP=np.shape(d)[0]
					trInput.append(d)
					trOutput.append(np.tile(vRegion[k]['Center'],(noP,1)))
			trInput=np.vstack((trInput))
			trOutput=np.vstack((trOutput))
		return trInput,trOutput

	def generateData(self,trData,trLabels,noC=1,noCERun=1):
		syntheticData=[]
		param = {}
		param['noCenter'] = noC #No of centers for kMean
		param['algo'] = 'kMean'
		param['nItr'] = 500 #No of iteration for kMean

		trInput,trOutput=self.createCEData(trData,trLabels,param)
 		########################################## Centroidencoder layerwise pre-training ##########################################
		dict1={}
		dict1['inputL'] = np.shape(trData)[1]
		dict1['outputL'] = np.shape(trData)[1]
		dict1['hL'] = self.params['hLayer']
		dict1['actFunc'] = self.params['actFunc']
		dict1['nItr']= self.params['noItrPre']*np.ones(len(dict1['actFunc'])).astype(int)
		dict1['errorFunc'] = self.params['errorFunc']

		cvThreshold,windowSize = 0,0;
		bottleneckPre=ae.BottleneckSAE(dict1)
		print('Network configuration:',dict1['inputL'],'-->',dict1['hL'],'-->',dict1['outputL'])
		print('Layer-wise pre-training the bottle-neck neural network.')
		#def train(self,iData,oData,valInput,valOutput,optimizationFuncName,cvThreshold,windowSize,nItr=10,weightPrecision=0,errorPrecision=0,verbose=False,freezeLayerFlag=True):
		bottleneckPre.train(trInput,trOutput,trInput,trOutput,self.params['optimizationFuncName'],cvThreshold,windowSize,self.params['noItrPre'])

		##########################################      Centroidencoder post-training     ##########################################
		for r in range(noCERun):
			dict_post={}
			dict_post['inputL'] = np.shape(trData)[1]
			dict_post['outputL'] = np.shape(trData)[1]
			dict_post['hL'] = self.params['hLayer']
			dict_post['actFunc'] = copy(self.params['actFunc'])
			dict_post['actFunc'].extend(['linear'])
			dict_post['nItr'] = self.params['noItrPost']
			dict_post['errorFunc'] = self.params['errorFunc']
			optimizationFuncNamePost = self.params['optimizationFuncName']
			bottleneckPost = ae.BottleneckAE(dict_post)
			windowSize = 0
			batchFlag = self.params['batchFlag']
			if r==0:
				bottleneckPost.netW = copy(bottleneckPre.netW)
			else:
				bottleneckPost.netW = self.parameters['preW']
			print('Post training bottle-neck neural network',str(dict_post['inputL']),str(dict_post['hL']),str(dict_post['outputL']))
			#def train(self,iData,oData,valInput,valOutput,optimizationFuncName,windowSize=0,nItr=10,dropPercentage=0,weightPrecision=0,errorPrecision=0,verbose=False):
			bottleneckPost.train(trInput,trOutput,trInput,trOutput,optimizationFuncNamePost,windowSize,self.params['noItrPost'])
			#Take the output of centroid-encoder and stack it with original trainind data to enlarge the trsining set.
			ceOutput = bottleneckPost.regenDWOStandardize(trData)[-1]
			syntheticData.append(np.hstack((ceOutput,trLabels)))
			trData = np.vstack((trData,ceOutput))
			trLabels = np.vstack((trLabels,trLabels))
			#With the added new ce output reclaculate centroids.
			trInput,trOutput=self.createCEData(trData,trLabels,param)
			#Save weight from pre-training
			self.parameters['preW'] = copy(bottleneckPost.netW)
		############################################### Centroidencoder post-training ###############################################
		del bottleneckPre,bottleneckPost,self.parameters['preW']
		syntheticData=np.vstack((syntheticData))
		return syntheticData[:,:-1],syntheticData[:,-1].reshape(-1,1)
        
	def fit(self,trData,trLabels):
		param = {}
		param['noCenter'] = 1 #No of centers for kMean
		param['algo'] = 'kMean'
		param['nItr'] = 500 #No of iteration for kMean
		trInput,trOutput=self.createCEData(trData,trLabels,param)
 		########################################## Centroidencoder layerwise pre-training ##########################################
		dict1={}
		dict1['inputL'] = np.shape(trData)[1]
		dict1['outputL'] = np.shape(trData)[1]
		dict1['hL'] = self.params['hLayer']
		dict1['actFunc'] = self.params['actFunc']
		dict1['nItr']= self.params['noItrPre']*np.ones(len(dict1['actFunc'])).astype(int)
		dict1['errorFunc'] = self.params['errorFunc']

		cvThreshold,windowSize = 0,0;
		bottleneckPre=ae.BottleneckSAE(dict1)
		print('Network configuration:',dict1['inputL'],'-->',dict1['hL'],'-->',dict1['outputL'])
		print('Layer-wise pre-training the bottle-neck neural network.')
		#def train(self,iData,oData,valInput,valOutput,optimizationFuncName,cvThreshold,windowSize,nItr=10,weightPrecision=0,errorPrecision=0,verbose=False,freezeLayerFlag=True):
		bottleneckPre.train(trInput,trOutput,trInput,trOutput,self.params['optimizationFuncName'],cvThreshold,windowSize,self.params['noItrPre'])

		##########################################      Centroidencoder post-training     ##########################################
		for r in range(self.params['repeatCE']):
			dict_post={}
			dict_post['inputL'] = np.shape(trData)[1]
			dict_post['outputL'] = np.shape(trData)[1]
			dict_post['hL'] = self.params['hLayer']
			dict_post['actFunc'] = copy(self.params['actFunc'])
			dict_post['actFunc'].extend(['linear'])
			dict_post['nItr'] = self.params['noItrPost']
			dict_post['errorFunc'] = self.params['errorFunc']
			optimizationFuncNamePost = self.params['optimizationFuncName']
			bottleneckPost = ae.BottleneckAE(dict_post)
			windowSize = 0
			batchFlag = self.params['batchFlag']
			if r==0:
				bottleneckPost.netW = copy(bottleneckPre.netW)
			else:
				bottleneckPost.netW = self.parameters['preW']
			print('Post training bottle-neck neural network',str(dict_post['inputL']),str(dict_post['hL']),str(dict_post['outputL']))
			#def train(self,iData,oData,valInput,valOutput,optimizationFuncName,windowSize=0,nItr=10,dropPercentage=0,weightPrecision=0,errorPrecision=0,verbose=False):
			bottleneckPost.train(trInput,trOutput,trInput,trOutput,optimizationFuncNamePost,windowSize,self.params['noItrPost'])
			if self.params['useSyntheticData'] == True:
				#Take the output of centroid-encoder and stack it with original trainind data to enlarge the trsining set.
				ceOutput = bottleneckPost.regenDWOStandardize(trData)[-1]
				trData = np.vstack((trData,ceOutput))
				trLabels = np.vstack((trLabels,trLabels))
				#With the added new ce output reclaculate centroids.
				trInput,trOutput=self.createCEData(trData,trLabels,param)
			#Save weight from pre-training
			self.parameters['preW'] = copy(bottleneckPost.netW)
		############################################### Centroidencoder post-training ###############################################

		###############################################   Linear Classifier Softmax   ###############################################
		print('Softmax classification starts.')
		#pdb.set_trace()
		newTrData=bottleneckPost.regenDWOStandardize(trData)[int(len(dict_post['actFunc'])/2)]
		#newTstData=bottleneckPost.regenDWOStandardize(tstData)[int(len(dict_post['actFunc'])/2)]

		#noTstData = np.shape(tstData)[0]
		optimizationFuncName = self.params['optimizationFuncName']
		windowSize = 50
		cvThreshold = 0
		noEpoch = 0
		batchFlag = self.params['batchFlag']
		batchSize = 0
		postBatchItr = self.params['noItrSoftmax']
		errThreshold = 0.025
		dict_softmax = {}
		dict_softmax['inputL'] = np.shape(newTrData)[1]
		dict_softmax['outputL'] = len(np.unique(trLabels))
		dict_softmax['hL'] = []
		dict_softmax['actFunc'] = []
		dict_softmax['nItr'] = []
		dict_softmax['errorFunc'] = 'CE'
		softmax = dc.deepClassifier(dict_softmax)
		softmax.postTrain(newTrData,trLabels,[],newTrData,trLabels,newTrData,trLabels,optimizationFuncName,windowSize,postBatchItr,batchFlag,batchSize,noEpoch,errThreshold)
		#Save weight from softmax layer
		self.parameters['softmaxW'] = copy(softmax.netW)
		predictedTrLabel,noErrorTr=softmax.test(newTrData,trLabels)
		print('No. of misclassification on training data after softmax=',noErrorTr)
		############################################### Linear Classifier Softmax ###############################################

		###############################################         Fine tuning       ###############################################
		#Now backpropagate error from softmax layer thru hidden layers.
		print('Starting fine tuning.')
		#pdb.set_trace()
		dict2 = {}
		dict2['inputL'] = np.shape(trData)[1]
		dict2['outputL'] = len(np.unique(trLabels))
		dict2['hL'] = dict_post['hL'][:int(np.ceil((len(dict_post['hL'])/2)))]
		dict2['actFunc'] = dict_post['actFunc'][:int(np.ceil((len(dict_post['actFunc'])/2)))]
		dict2['nItr'] = []
		dict2['errorFunc'] = 'CE'
		finetune=dc.deepClassifier(dict2)
		finetune.preTrainingDone = True

		for l in range(int(len(dict_post['actFunc'])/2)):
			finetune.netW.append(copy(bottleneckPost.netW[l]))
		finetune.netW.append(copy(softmax.netW[0]))
		noEpoch=0
		windowSize=50
		batchFlag=False
		postBatchItr=self.params['noItrFinetune']
		postItr=0
		errThreshold=1.0e-12
		optimizationFuncName='scg'
		finetune.postTrain(trData,trLabels,[],trData,trLabels,trData,trLabels,optimizationFuncName,windowSize,postBatchItr,batchFlag,batchSize,noEpoch,errThreshold)
		#Save weight from fine layer
		self.parameters['finetuneW'] = copy(finetune.netW)
		predictedTrLabel,noErrorTr1=finetune.test(trData,trLabels)
		print('No. of misclassification on training data after fine tuning=',noErrorTr1)
		trAccuracy=(1-noErrorTr1/len(trData))*100
		print('Accuracy on traing data = ','{0:.2f}'.format(trAccuracy))
		############################################### Fine tuning ###############################################

		self.testClassifier=dc.deepClassifier(dict2)
		self.testClassifier.preTrainingDone=True
		self.testClassifier.netW = copy(finetune.netW)
		return self.parameters

	def classify(self,data):
		'''
		data: test data
		'''
		lOut=[data]
		lLength=len(self.testClassifier.netW)

		for j in range(lLength):
			d = np.dot(lOut[-1],self.testClassifier.netW[j][1:,:])+self.testClassifier.netW[j][0]#first row in the weight is the bias
			#Take the activation function from the dictionary and apply it
			lOut.append(self.feval('self.'+self.testClassifier.actFunc[j],d) if j<lLength-1 else d)
		fOut = self.calcLogProb(lOut[-1])
		predictedTstLabel=(np.argmax(fOut,axis=1)).reshape(-1,1)
		return predictedTstLabel
