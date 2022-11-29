from calcom.classifiers._abstractclassifier import AbstractClassifier

class EnsembleClassifier(AbstractClassifier):
	def __init__(self,ensembleMethod='bagging',model=[],pseudoLoss=False):
		#ensembleMethod: type of ensembling: 'bagging','standardensemble','adaboost','gradientboosting' etc
		#model: an array of learner/classifier.
		'''
		Setup default parameters
		'''
		self.params = {}
		self.testClassifier = []
		self.models = []
		if len(model) != 0:#when model is passed
			self.params['learner'] = model
			self.params['noModel'] = len(model)
		else:#when no model is passed initialize with centroidencoder
			import calcom
			self.params['noModel'] = 5#default is 5
			self.params['learner'] = []
			for i in range (5):
				ce = calcom.classifiers.CentroidencoderClassifier()
				ce.params['hLayer'] = [100,25,100]
				ce.params['actFunc'] = ['rectl','rectl','rectl']
				ce.params['noItrPre'] = 10
				ce.params['noItrPost'] = 20
				ce.params['noItrSoftmax'] = 5
				ce.params['noItrFinetune'] = 5
				self.params['learner'].append(ce)

		self.params['ensembleMethod'] = ensembleMethod
		self.params['pseudoLoss'] = pseudoLoss
		self.params['dataDim'] = 0
		self.params['classList'] = []
		self.params['noClass'] = 0
		self.params['noSamples'] = 0
		self.params['boostW'] = []
		self.params['alpha'] = []
	#

	@property
	def _is_native_multiclass(self):
		# this is an imperfect solution, if different model types are included.
		# todo in distant future: separate classifiers 
		# and ensemble tools into separate categories.
		return self.params['learner'][0]._is_native_multiclass
	#
	
	def _is_ensemble_method(self):
		return True
	#

	def feval(self,fName,*args):
		return eval(fName)(*args)

	def returnDuplicateElement(self,L):
		import numpy as np
		u, c = np.unique(L, return_counts=True)
		return u[np.where(c>1)]

	def predictByVoting(self,modelsPredictedValLabel):
		#Each row contains prediction label from 'm' different classifiers.
		#Majority voting will be used to get the final prediction
		#pdb.set_trace()
		import numpy as np
		predictedLabels = []
		for i in range(len(modelsPredictedValLabel)):
			L=list(modelsPredictedValLabel[i,:])
			predictedLabels.append(max(L,key=L.count))
		predictedLabels = np.hstack((predictedLabels)).reshape(-1,1)
		return predictedLabels

	def predictWithBoosting(self,data):
		#H(x)=sign(alpha_1*h_1(x)+alpha_2*h_2(x)+...+alpha_n*h_n(x))
		#pdb.set_trace()
		import numpy as np
		if 'LinearSVC' in str(self.params['learner'][0]) or 'SSVM' in str(self.params['learner'][0]):
			scores = np.zeros([len(data),1])
			pLabels = np.zeros([len(data),1])
			for c in range(len(self.models)):
				scores = scores + self.params['alpha'][c] * self.models[c].decision_function(data).reshape(-1,1)
			pLabels[np.where(scores>=0)[0],:] = 1
		elif 'NeuralnetworkClassifier' in str(self.params['learner'][0]) or 'CentroidencoderClassifier' in str(self.params['learner'][0]):
			scores = np.zeros([len(data),self.params['noClass']])
			pLabels = np.zeros([len(data),1])
			#pdb.set_trace()
			for c in range(len(self.models)):
				scores = scores + self.params['alpha'][c] * self.models[c].decision_function(data)
			pLabels = (np.argmax(scores,axis=1)).reshape(-1,1)
		return pLabels

	def returnNoError(self,aLabel,pLabel):
		misClassifiedD = []
		if aLabel.ndim == 1:
			aLabel.reshape(-1,1)
		noData = len(aLabel)
		misClassifiedD = [aLabel[i] for i in range(noData) if aLabel[i] != pLabel[i]]
		return len(misClassifiedD)

	def returnSuccessErrorIndex(self,aLabel,pLabel):
		if aLabel.ndim == 1:
			aLabel.reshape(-1,1)
		noData = len(aLabel)
		errorIndices = [i for i in range(noData) if aLabel[i] != pLabel[i]]
		successIndices = [i for i in range(noData) if aLabel[i] == pLabel[i]]
		return successIndices,errorIndices

	def generateBaggingIndices(self,label):
		#This function will return sample indices using sampling(uniform) with replacement
		indices = []
		'''
		indices = [np.random.choice(len(label),len(label)) for i in range(self.params['noModel'])]
		return indices

		#pdb.set_trace()
		'''
		import numpy as np
		#stratified bagging, useful for imbalanced data
		i = 0
		for c in np.unique(label):
			tmpIndices = [np.random.choice(np.where(label==c)[0],len(np.where(label==c)[0])) for i in range(self.params['noModel'])]
			tmpIndices = np.vstack((tmpIndices))
			if i == 0:
				indices = tmpIndices
			else:
				indices = np.hstack((indices,tmpIndices))
			i += 1
		return indices

	def sampling(self,probDistribution,nSample):
		import numpy as np
		return np.random.choice(nSample,nSample,p=probDistribution)

	def runEnsembleMethods(self,trData,trLabel):
		if self.params['ensembleMethod'].upper() == 'BAGGING':
			#pLabel,pVal = self.runBagging(trData,trLabel)
			pLabel = self.runBagging(trData,trLabel)
		elif self.params['ensembleMethod'].upper() == 'ADABOOST':
			pLabel = self.runAdaBoost(trData,trLabel)
		elif self.params['ensembleMethod'].upper() == 'GRADIENTBOOSTING':
			print('TBD')
		elif self.params['ensembleMethod'].upper() == 'STANDARDENSEMBLE':
			#pLabel,pVal = self.runStandardEnsemble(trData,trLabel)
			pLabel = self.runStandardEnsemble(trData,trLabel)
		else:
			print('Wrong ensemble method')
		noErrorTr = self.returnNoError(trLabel,pLabel)
		print('No. of misclassification on training data =',noErrorTr)
		trAccuracy = (1-noErrorTr/len(trData))*100
		print('Success rate on traing data = ','{0:.2f}'.format(trAccuracy))

	def runAdaBoost(self,trData,trLabel):
		if 'LinearSVC' in str(self.params['learner'][0]):
			pLabel = self.boostingWithLinearSVM(trData,trLabel)
		elif 'SSVMClassifier' in str(self.params['learner'][0]):
			pLabel = self.boostingWithLinearSVM(trData,trLabel)
		elif 'NeuralnetworkClassifier' in str(self.params['learner'][0]):
			pLabel = self.boostingWithANN(trData,trLabel)
		elif 'CentroidencoderClassifier' in str(self.params['learner'][0]):
			pLabel = self.boostingWithCE(trData,trLabel)
		else:
			print('Unknown base classifier.')
		return pLabel

	def boostingWithANN(self,trData,trLabel):
		from copy import deepcopy
		import numpy as np

		nSample = len(trData)
		self.params['boostW'] = (1/nSample)*np.ones(nSample).reshape(-1,1)#Initialize weights uniformly for training sample
		#pdb.set_trace()
		for m in range(len(self.params['learner'])):#iterate thru given base model
			prob = self.params['boostW'][:,-1]
			sampleIndices = self.sampling(prob,nSample)
			newTrData,newTrLabel = trData[sampleIndices,:],trLabel[sampleIndices,:]
			self.models.append(deepcopy(self.params['learner'][m].fit(newTrData,newTrLabel.flatten())))
			#Calculate the loss
			if not(self.params['pseudoLoss']):
				#Calculate the misclassfication on training data
				predictedLabels = self.models[-1].predict(trData).reshape(-1,1)
				successIndices,errorIndices = self.returnSuccessErrorIndex(trLabel,predictedLabels)
				if len(errorIndices)==0:#This means no error for this model
					print('Model error is 0, assigning a small number in epsilon.')
					epsilon = np.finfo(np.float64).eps #assign a samll number.
				else:
					epsilon = np.sum(self.params['boostW'][errorIndices,-1])
			else:
				pLoss = self.models[-1].pseudoLoss(trData,trLabel)
				epsilon = np.sum(prob*pLoss)
			#Using the lecture on boosting from "https://www.youtube.com/watch?v=gmok1h8wG-Q"
			alpha = 0.5*np.log((1-epsilon)/epsilon)#This is the voting power of the current classifier
			self.params['alpha'].append(alpha)
			#Check for termination.

			if m > 0 and m < len(self.params['learner']) :#when m>0 we combined two classifiers. Now check the error on training data using the composition of classifiers.
			#If no of misclassification on training data is zero the stop boosting otherwise proceed to next iteration.
				pLabel = self.predict(trData).reshape(-1,1)
				noErrorTr = self.returnNoError(trLabel,pLabel)
				'''
				if noErrorTr == 0:
					print('Misclassification on training data is 0, No of classifier: ',m+1 ,'stopping boosting...')
					return pLabel
				'''

			#Now update weights. The weights of correctly classified samples will be down graded and the weights of the misclassified
			#samples will be upgraded. This weight list will be used as probability distribution to resample training data in next iteration.
			newW = deepcopy(self.params['boostW'][:,-1].reshape(-1,1))
			if not(self.params['pseudoLoss']):
				newW[successIndices] *= 0.5*(1/(1-epsilon))
				newW[errorIndices] *= 0.5*(1/epsilon)
			else:
				#newW *= epsilon #These epsilons are the probabilities and their sum can't be greater than 1.
				newW *= pLoss.reshape(-1,1) #These epsilons are the probabilities and their sum can't be greater than 1.
			'''
			#Using the paper "https://pdfs.semanticscholar.org/0d7d/bd8503a9fe61c8e02465de2fa327e4d89c05.pdf"
			#alpha = epsilon/(1-epsilon)#This is the voting power of the current classifier
			#Now update weights. The weights of correctly classified samples will be down graded and the weights of the misclassified
			#samples will be upgraded. This weight list will be used as probability distribution to resample training data in next iteration.
			newW = deepcopy(self.params['boostW'][:,-1].reshape(-1,1))
			newW[successIndices] *= alpha
			'''
			newW /= np.sum(newW)#Normalize the new weights to make sure they sum to 1.
			self.params['boostW'] = np.concatenate((self.params['boostW'],newW),axis=1)

		print('Maximum iteration is reached. Stopping boosting...')
		return pLabel

	def boostingWithCE(self,trData,trLabel):
		from copy import deepcopy
		import numpy as np

		nSample = len(trData)
		self.params['boostW'] = (1/nSample)*np.ones(nSample).reshape(-1,1)#Initialize weights uniformly for training sample
		#pdb.set_trace()
		for m in range(len(self.params['learner'])):#iterate thru given base model
			prob = self.params['boostW'][:,-1]
			sampleIndices = self.sampling(prob,nSample)
			newTrData,newTrLabel = trData[sampleIndices,:],trLabel[sampleIndices,:]
			self.models.append(deepcopy(self.params['learner'][m].fit(newTrData,newTrLabel.flatten())))
			#Calculate the loss
			if not(self.params['pseudoLoss']):
				#Calculate the misclassfication on training data
				predictedLabels = self.models[-1].predict(trData).reshape(-1,1)
				successIndices,errorIndices = self.returnSuccessErrorIndex(trLabel,predictedLabels)
				if len(errorIndices)==0:#This means no error for this model
					print('Model error is 0, assigning a small number in epsilon.')
					epsilon = np.finfo(np.float64).eps #assign a samll number.
				else:
					epsilon = np.sum(self.params['boostW'][errorIndices,-1])
			else:
				pLoss = self.models[-1].pseudoLoss(trData,trLabel)
				epsilon = np.sum(prob*pLoss)
			#Using the lecture on boosting from "https://www.youtube.com/watch?v=gmok1h8wG-Q"
			alpha = 0.5*np.log((1-epsilon)/epsilon)#This is the voting power of the current classifier
			self.params['alpha'].append(alpha)
			#Check for termination.
			if m > 0 and m < len(self.params['learner']) :#when m>0 we combined two classifiers. Now check the error on training data using the composition of classifiers.
			#If no of misclassification on training data is zero the stop boosting otherwise proceed to next iteration.
				pLabel = self.predict(trData).reshape(-1,1)
				noErrorTr = self.returnNoError(trLabel,pLabel)
				'''
				if noErrorTr == 0:
					print('Misclassification on training data is 0, No of classifier: ',m+1 ,'stopping boosting...')
					return pLabel
				'''
			#Now update weights. The weights of correctly classified samples will be down graded and the weights of the misclassified
			#samples will be upgraded. This weight list will be used as probability distribution to resample training data in next iteration.
			newW = deepcopy(self.params['boostW'][:,-1].reshape(-1,1))
			if not(self.params['pseudoLoss']):
				newW[successIndices] *= 0.5*(1/(1-epsilon))
				newW[errorIndices] *= 0.5*(1/epsilon)
			else:
				#newW *= epsilon #These epsilons are the probabilities and their sum can't be greater than 1.
				newW *= pLoss.reshape(-1,1) #These epsilons are the probabilities and their sum can't be greater than 1.
			'''
			#Using the paper "https://pdfs.semanticscholar.org/0d7d/bd8503a9fe61c8e02465de2fa327e4d89c05.pdf"
			#alpha = epsilon/(1-epsilon)#This is the voting power of the current classifier
			#Now update weights. The weights of correctly classified samples will be down graded and the weights of the misclassified
			#samples will be upgraded. This weight list will be used as probability distribution to resample training data in next iteration.
			newW = deepcopy(self.params['boostW'][:,-1].reshape(-1,1))
			newW[successIndices] *= alpha
			'''
			newW /= np.sum(newW)#Normalize the new weights to make sure they sum to 1.
			self.params['boostW'] = np.concatenate((self.params['boostW'],newW),axis=1)

		print('Maximum iteration is reached. Stopping boosting...')
		return pLabel


	def boostingWithLinearSVM(self,trData,trLabel):
		#pdb.set_trace()
		import numpy as np

		nSample = len(trData)
		self.params['boostW'] = (1/nSample)*np.ones(nSample).reshape(-1,1)#Initialize weights of each training samples uniformly
		for m in range(len(self.params['learner'])):#iterate thru all the model
			prob = self.params['boostW'][:,-1]
			sampleIndices = self.sampling(prob,nSample)
			newTrData,newTrLabel = trData[sampleIndices,:],trLabel[sampleIndices,:]
			self.models.append(deepcopy(self.params['learner'][m].fit(newTrData,newTrLabel.flatten())))
			#Calculate the misclassfication on training data
			predictedLabels = self.models[-1].predict(trData).reshape(-1,1)
			successIndices,errorIndices = self.returnSuccessErrorIndex(trLabel,predictedLabels)
			if len(errorIndices)==0:#This means no error for this model
				print('Model error is 0, assigning a small number in epsilon.')
				epsilon = np.finfo(np.float64).eps#assign a samll number.
			else:
				epsilon = np.sum(self.params['boostW'][errorIndices,-1])#Calculate error for current classiifer based on misclassification
			#Using the lecture on boosting from "https://www.youtube.com/watch?v=gmok1h8wG-Q"
			alpha = 0.5*np.log((1-epsilon)/epsilon)#This is the voting power of the current classifier
			self.params['alpha'].append(alpha)
			#Check for termination.
			if m > 0 and m < len(self.params['learner']) :#when m>0 we combined two classifiers. Now check the error on training data using the composition of classifiers.
			#If no of misclassification on training data is zero the stop boosting otherwise proceed to next iteration.
				#pdb.set_trace()
				pLabel = self.predict(trData).reshape(-1,1)
				noErrorTr = self.returnNoError(trLabel,pLabel)
				'''
				if noErrorTr == 0:
					print('Misclassification on training data is 0, No of classifier: ',m+1 ,'stopping boosting...')
					return pLabel
				'''
			#Now update weights. The weights of correctly classified samples will be down graded and the weights of the misclassified
			#samples will be upgraded. This weight list will be used as probability distribution to resample training data in next iteration.
			newW = deepcopy(self.params['boostW'][:,-1].reshape(-1,1))
			newW[successIndices] *= 0.5*(1/(1-epsilon))
			newW[errorIndices] *= 0.5*(1/epsilon)
			'''
			#Using the paper "https://pdfs.semanticscholar.org/0d7d/bd8503a9fe61c8e02465de2fa327e4d89c05.pdf"
			#alpha = epsilon/(1-epsilon)#This is the voting power of the current classifier
			#Now update weights. The weights of correctly classified samples will be down graded and the weights of the misclassified
			#samples will be upgraded. This weight list will be used as probability distribution to resample training data in next iteration.
			newW = deepcopy(self.params['boostW'][:,-1].reshape(-1,1))
			newW[successIndices] *= alpha
			'''
			newW /= np.sum(newW)#Normalize the new weights to make sure they sum to 1.
			self.params['boostW'] = np.concatenate((self.params['boostW'],newW),axis=1)

		print('Maximum iteration is reached. Stopping boosting...')
		pLabel = self.predict(trData).reshape(-1,1)
		return pLabel

	def runBagging(self,trData,trLabel):
		import numpy as np

		indexMat = self.generateBaggingIndices(trLabel) #Each row of the indexMat contains sample indices from original data. Indices in each row
		#corresponds to a bagging training set. No of rows corresponds to number of models to be trained.
		#now loop thru all the bagging training index set and train seperate model
		modelsPredictedLabels = []
		modelsPredictedValues = []
		for m in range(len(indexMat)):
			print('Training model ',m+1)
			baggingData = trData[indexMat[m],:]
			baggingLabel = trLabel[indexMat[m],:]
			if 'LinearSVC' in str(self.params['learner'][0]):
				baggingLabel = baggingLabel.flatten()
			self.models.append(deepcopy(self.params['learner'][m].fit(baggingData,baggingLabel)))
			predictedLabels = self.models[-1].predict(trData).reshape(-1,1)
			modelsPredictedLabels.append(predictedLabels)
			#modelsPredictedValues.append(predictedValues)
		modelsPredictedLabel = np.hstack((modelsPredictedLabels)) #Each row contains predicted label for a validation sanmple from different models.
		#modelsPredictedValues = np.hstack((modelsPredictedValues)) #Each row contains predicted value for a validation sanmple from different models.
		finalPredictedLabels = self.predictByVoting(modelsPredictedLabel) #Use voting to get the final prediction.
		#modelsPredictedValues = np.mean(modelsPredictedValues,axis=1)
		return finalPredictedLabels

	def runStandardEnsemble(self,trData,trLabel):
		import numpy as np

		modelsPredictedLabels = []
		modelsPredictedValues = []
		for m in range(len(self.params['learner'])):
			print('Training model ',m+1)
			self.models.append(deepcopy(self.params['learner'][m].fit(trData,trLabel)))
			predictedLabels = self.models[-1].predict(trData).reshape(-1,1)
			modelsPredictedLabels.append(predictedLabels)
		modelsPredictedLabel = np.hstack((modelsPredictedLabels)) #Each row contains predicted label for a validation sanmple from different models.
		finalPredictedLabels = self.predictByVoting(modelsPredictedLabel) #Use voting to get the final prediction.
		#modelsPredictedValues = np.mean(modelsPredictedValues,axis=1)
		return finalPredictedLabels

	def _fit(self,trData,trLabel):
		#data: A [m x n] array where m = no. of samples, n = no. of features/dimensions
		#label: A [m x 1] array where each element is an integer.
		import numpy as np

		# internal_labels = self._process_input_labels(trLabel)
		internal_labels = trLabel

		self.params['dataDim'] = np.shape(trData)[1]
		self.params['classList'] = self._label_info['unique_labels_mapped']
		self.params['noClass'] = len(self.params['classList'])
		self.params['noSamples'] = len(trData)
		# self.runEnsembleMethods(trData,trLabel)
		self.runEnsembleMethods(trData,internal_labels)
		return self

	def _predict(self,data):
		#data: test/validation data
		import numpy as np

		modelsPredictedLabels = []
		modelsPredictedValues = []
		#pdb.set_trace()
		if 'BOOST' in self.params['ensembleMethod'].upper():
			finalPredictedLabels = self.predictWithBoosting(data)
		else:
			for m in range(self.params['noModel']):
				predictedLabels = self.models[m].predict(data).reshape(-1,1)
				modelsPredictedLabels.append(predictedLabels)
				#modelsPredictedValues.append(predictedValues)
			modelsPredictedLabel = np.hstack((modelsPredictedLabels)) #Each row contains predicted label for a validation sanmple from different models.
			#modelsPredictedValues = np.hstack((modelsPredictedValues)) #Each row contains predicted value for a validation sanmple from different models.
			finalPredictedLabels = self.predictByVoting(modelsPredictedLabel) #Use voting to get the final prediction.
			#modelsPredictedValues = np.mean(modelsPredictedValues,axis=1)
		#
		# pred_labels = self._process_output_labels(finalPredictedLabels)
		#
		# return pred_labels
		return finalPredictedLabels
	#

	def visualize(self,*args):
		pass

	def __str__(self):
		return super().__str__()

	def __repr__(self):
		return super().__repr__()
