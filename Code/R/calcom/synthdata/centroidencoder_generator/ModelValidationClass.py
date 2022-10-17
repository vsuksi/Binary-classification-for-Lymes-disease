import pdb
import pandas
from copy import copy
import numpy as np
from utilityDBN import confusionMatrix,multiclassConfusionMatrix,standardizeData,unStandardizeData

class ModelValidationClass:
	def __init__(self,data,label,model,expDetails={},minorityClass=[]):
		#data: A [m x n] array where m = no. of samples, n = no. of features/dimensions
		#label: A [m x 1] array where each element is an integer.
		self.classList = np.unique(label)
		self.noClass = len(self.classList)
		self.dataDim = np.shape(data)[1]
		self.noSamples = len(data)
		self.data = data
		self.label = label
		self.model = model		
		self.syntheticData = []
		self.syntheticLabel = []
		self.splitRatio = 0.8

		if expDetails == {}:#when no dictionary is passed
			self.expType = 'HoldoutCrossValidation'
			self.splitRatio = 0.8
			self.repetition = 1
			self.noFold = None
			self.useSyntheticData = False
		else:
			self.expType = expDetails['expType']
			if self.expType in ['HoldoutCrossValidation','RandomSubSampling']:
				self.noFold = None
			elif self.expType == 'LOO':
				self.noFold = self.noSamples
			else:
				self.noFold = expDetails['no_fold']
			if expDetails['repetition'] == None:
				self.repetition = 1
			else:
				self.repetition = expDetails['repetition']
			if self.expType == 'LOO' or 'CrossValidation':
				self.splitRatio = None
			else:
				self.splitRatio = expDetails['split_ratio']
			if 'useSyntheticData' in expDetails.keys():
				self.useSyntheticData = expDetails['useSyntheticData']
				self.minorityClass = minorityClass
			
	def splitData(self,data,label):
		#This method will split the dataset into training and test set based on the split ratio.
		#Training and test set will have data from each class according to the split ratio.
		#First hold the data of different classes in different variables.
		train_set =[]
		test_set =[]
		labeled_data = np.hstack((data,label))
		sorted_data = labeled_data[np.argsort(labeled_data[:,-1])]#sorting based on the numeric label.
		first_time = 'Y'
		for classes in self.classList:
			temp_class = np.array([sorted_data[i] for i in range(self.noSamples) if sorted_data[i,-1] == classes])
			np.random.shuffle(temp_class)#Shuffle the data so that we'll get variation in each run
			tr_samples = np.floor(len(temp_class)*self.splitRatio)
			tst_samples = len(temp_class) - tr_samples
			if(first_time == 'Y'):
				train_set = temp_class[:int(tr_samples),]
				test_set = temp_class[-int(tst_samples):,]
				first_time = 'N'
			else:
				train_set = np.vstack((train_set,temp_class[:int(tr_samples),]))
				test_set = np.vstack((test_set,temp_class[-int(tst_samples):,]))
		return train_set[:,:-1],train_set[:,-1].reshape(-1,1),test_set[:,:-1],test_set[:,-1].reshape(-1,1)
		
	def segmentData(self,data,label):
		#This function will partition the data into k-folds stratified sampling.
		#It's an absolute requirement to make the labels numeric starting from 0.
		#label should be an [m x 1] array
		dataSegment = {}
		classData = {}
		labeledData = np.hstack((data,label))
		noClass = np.unique(label)
		#pdb.set_trace()
		for c in range(len(noClass)):
			classKey = 'class'+str(c)
			classData[classKey] = labeledData[np.where(label==c)[0],:]
		for fold in range(self.noFold):
			foldKey = 'fold'+str(fold+1)
			foldData = []
			for c in range(len(noClass)):
				classKey = 'class'+str(c)
				noSample = int(len(classData[classKey])/self.noFold)
				sIndex = fold*noSample
				eIndex = sIndex + noSample
				if fold+1 == self.noFold:
					foldData.append(classData[classKey][sIndex:,:])
				else:
					foldData.append(classData[classKey][sIndex:eIndex,:])
			dataSegment[foldKey] = np.vstack((foldData))				
		return dataSegment
		
	def standardizeData(self,data,mu=[],std=[]):
		#data: a m x n matrix where m is the no of observations and n is no of features
		#if any(mu) == None and any(std) == None:
		if not(len(mu) and len(std)):
			#pdb.set_trace()
			std = np.std(data,axis=0)
			mu = np.mean(data,axis=0)
			std[np.where(std==0)[0]] = 1.0 #This is for the constant features.
			standardizeData = (data - mu)/std
			return mu,std,standardizeData
		else:
			standardizeData = (data - mu)/std
			return standardizeData
			
	def extractData(self,data,label,extractLabel):	
		indices=np.where(label==extractLabel)[0]
		return data[indices,:],label[indices,:]
			
	def unStandardizeData(self,data,mu,std):
		return std * data + mu
		
	def feval(self,fName,*args):
		return eval(fName)(*args)
		
	def HoldoutCrossValidation(self,data,label,model):
		for itr in range(self.repetition):
			trData,trLabel,valData,valLabel = self.splitData(data,label)			
			mu,std,standardizeTrData = self.standardizeData(trData)			
			modelParam = model.fit(standardizeTrData,trLabel)			
			standardizeValData = self.standardizeData(valData,mu,std) #validation data standardization using mu and std from training data
			predictedValLabel = model.classify(standardizeValData)			
		return predictedValLabel,modelParam
		
	def RandomSubSampling(self,data,label,model):
		allModelParams = []
		allPredictedLabels = []
		for itr in range(self.repetition):
			print('Repetition ',itr+1)
			print('=====================================================')
			trData,trLabel,valData,valLabel = self.splitData(data,label)			
			mu,std,standardizeTrData = self.standardizeData(trData)			
			modelParam = model.fit(standardizeTrData,trLabel)			
			standardizeValData = self.standardizeData(valData,mu,std) #validation data standardization using mu and std from training data
			predictedValLabel = model.classify(standardizeValData)
			allModelParams.append(modelParam)
			allPredictedLabels.append(predictedValLabel)
		return allPredictedLabels,allModelParams
		
	def CrossValidation(self,data,label,model,syntheticData=[],syntheticLabel=[]):
		allModelParams = []
		allPredictedLabels = []
		allActualLabels = []
		foldData = self.segmentData(data,label)
		allBSR = []
		allSuccessRate = []
		results={}
		for fold in range(self.noFold):
			print('Starting fold: ',fold+1)
			print('=========================================================')
			trData = []
			trLabel = []
			valData = []
			valLabel = []			
			for i in range(self.noFold):
				if i+1==fold+1:
					valData = foldData['fold'+str(fold+1)][:,:-1]
					valLabel = foldData['fold'+str(fold+1)][:,-1].reshape(-1,1)
				else:
					trData.append(foldData['fold'+str(i+1)][:,:-1])
					trLabel.append(foldData['fold'+str(i+1)][:,-1].reshape(-1,1))					
			trData = np.vstack((trData))
			trLabel = np.vstack((trLabel))			
			if self.useSyntheticData == True:
				#Add synthetic data only in model training. Don't use it in testing				
				#Now eaxtract the minority data and enlarge its size by using CE data genetator.
				#pdb.set_trace()
				for c in self.minorityClass:#Loop thru all the minority classes
					minorityData = trData[np.where(trLabel==c)[0],:]
					Labels = c*np.ones([len(minorityData),1])
					mu1,std1,standardizeMinorityData = standardizeData(minorityData)
					syntheticData,syntheticLabel = model.generateData(standardizeMinorityData,Labels,int(len(Labels)/2),1)
					syntheticData = unStandardizeData(syntheticData,mu1,std1)
					trData = np.vstack((np.vstack((trData)),syntheticData))
					trLabel = np.vstack((np.vstack((trLabel)),syntheticLabel))
			mu,std,standardizeTrData = self.standardizeData(trData)
			modelParam = model.fit(standardizeTrData,trLabel)
			standardizeValData = self.standardizeData(valData,mu,std)
			predictedValLabel = model.classify(standardizeValData)			
			allModelParams.append(modelParam)
			allPredictedLabels.append(predictedValLabel)
			if len(np.unique(trLabel))==2:
				sr,bsr,TPR,TNR=confusionMatrix(valLabel,predictedValLabel,classLabelName=['control','shedder5hr'],display_flag='N')				
			else:#for multiclass
				sr,bsr=multiclassConfusionMatrix(valLabel,predictedValLabel,classLabelName=['control','Symptomatic','Asymptomatic'],display_flag='N')
			allBSR.append(bsr)
			allSuccessRate.append(sr)
		#print('Average balanced success rate = ','{0:.2f}'.format(np.mean(allBSR)),'(+/-)','{0:.2f}'.format(np.std(allBSR)))
		results['allSR']=allSuccessRate
		results['allBSR']=allBSR
		return allPredictedLabels,allModelParams,results
		
	def runValidation(self):
		if self.useSyntheticData != True:
			predictedLabel,modelParam,results = self.feval('self.'+self.expType,self.data,self.label,self.model)
		else:
			predictedLabel,modelParam,results = self.feval('self.'+self.expType,self.data,self.label,self.model,self.syntheticData,self.syntheticLabel)
		return predictedLabel,modelParam,results
