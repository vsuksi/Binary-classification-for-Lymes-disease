import pdb
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from collections import deque
from calcom.classifiers._abstractclassifier import AbstractClassifier

class NeuralnetworkClassifierPyTorch(nn.Module,AbstractClassifier):

	def __init__(self,netConfig={}):
		super(NeuralnetworkClassifierPyTorch, self).__init__()
		#first do the default set up of parameters in a dictionary
		self.params = {}
		self.params['hLayer'] = [75,40]
		self.params['hActFunc'] = ['lrelu','lrelu']
		self.params['earlyStop'] = True
		self.params['earlyStopWindow'] = 50
		self.params['errorCutoff'] = 0.025
		self.params['noItrPre'] = 10
		self.params['noItrPost'] = 40
		self.params['l1Penalty'], self.params['l2Penalty'] = 0.0,0.0
		self.params['optimizationFunc'] = 'Adam'
		self.params['learningRate'] = 0.001
		self.params['miniBatchSize'] = 32
		self.params['momentum'] = 0.8 # this will be used only for SGD. Adam doesn't need this.
		self.params['standardizeFlag'] = True
		
		#now check if the dictionary is passed.
		if len(netConfig.keys()) != 0:			
			self.params['hLayer'],self.params['hLayerPost'] = deepcopy(netConfig['hL']),deepcopy(netConfig['hL'])
			if 'earlyStop' in netConfig.keys():
				self.params['earlyStop'] = netConfig['earlyStop']
				self.params['earlyStopWindow'] = 50
				self.params['errorCutoff'] = 0.025
			if 'optimizationFunc' in netConfig.keys(): self.params['optimizationFunc'] = netConfig['optimizationFunc']
			if 'learningRate' in netConfig.keys(): self.params['learningRate'] = netConfig['learningRate']
			if 'miniBatchSize' in netConfig.keys(): self.params['miniBatchSize'] = netConfig['miniBatchSize']
			if 'standardizeFlag' in netConfig.keys(): self.params['standardizeFlag'] = netConfig['standardizeFlag']
			if 'l1Penalty' in netConfig.keys(): self.params['l1Penalty'] = netConfig['l1Penalty']
			if 'l2Penalty' in netConfig.keys(): self.params['l2Penalty'] = netConfig['l2Penalty']
			if 'linearDecoder' in netConfig.keys(): params['linearDecoder'] = netConfig['linearDecoder']
		
		#internal variables
		self.inputDim,self.nClass = None,None
		self.currentHL = None
		self.epochError = []
		self.trMu = []
		self.trSd = []
		self.tmpPreHActFunc = []
		self.preTrW = []
		self.preTrB = []
		self.device = None
		self.classifier = None
		self.fineTuneW = []
		self.fineTuneB = []
		self.nLayers = len(self.params['hLayer'])
		self.hLayer = self.params['hLayer']
		self.hActFunc = self.params['hActFunc']
		self.l1Penalty,self.l2Penalty = self.params['l1Penalty'],self.params['l2Penalty']
		self.earlyStop = self.params['earlyStop']
		self.earlyStopWindow, self.errorCutoff= self.params['earlyStopWindow'],self.params['errorCutoff']

	@property
	def _is_native_multiclass(self):
		return True
	#
	@property
	def _is_ensemble_method(self):
		return False

	def returnParams(self):
		for p in self.params.keys():
			print(p,':',self.params[p])

	def initParams(self,*fargs,**params):

		if 'hLayer' in params.keys(): self.params['hLayer'] = params.pop('hLayer')
		if 'hActFunc' in params.keys(): self.params['hActFunc'] = params.pop('hActFunc')
		if 'optimizationFunc' in params.keys(): self.params['optimizationFunc'] = params.pop('optimizationFunc')
		if 'momentum' in params.keys(): self.params['momentum'] = params.pop('momentum')
		if 'learningRate' in params.keys(): self.params['learningRate'] = params.pop('learningRate')
		if 'miniBatchSize' in params.keys(): self.params['miniBatchSize'] = params.pop('miniBatchSize')
		if 'standardizeFlag' in params.keys(): self.params['standardizeFlag'] = params.pop('standardizeFlag')
		if 'earlyStop' in params.keys(): self.params['earlyStop'] = params.pop('earlyStop')
		if 'earlyStopWindow' in params.keys(): self.params['earlyStopWindow'] = params.pop('earlyStopWindow')
		if 'errorCutoff' in params.keys(): self.params['errorCutoff'] = params.pop('errorCutoff')		
		if 'noItrPre' in params.keys(): self.params['noItrPre'] = params.pop('noItrPre')
		if 'noItrPost' in params.keys(): self.params['noItrPost'] = params.pop('noItrPost')
		if 'l1Penalty' in params.keys(): self.params['l1Penalty'] = params.pop('l1Penalty')
		if 'l2Penalty' in params.keys(): self.params['l2Penalty'] = params.pop('l2Penalty')
		
		self.nLayers = len(self.params['hLayer'])
		self.hLayer = self.params['hLayer']
		self.hActFunc = self.params['hActFunc']
		self.l1Penalty,self.l2Penalty = self.params['l1Penalty'],self.params['l2Penalty']
		self.earlyStop = self.params['earlyStop']
		self.earlyStopWindow, self.errorCutoff= self.params['earlyStopWindow'],self.params['errorCutoff']

	def initNet(self,input_size,hidden_layer):
		#pdb.set_trace()
		self.hidden = nn.ModuleList()
		# Hidden layers
		if len(hidden_layer)==1:
			self.hidden.append(nn.Linear(input_size,hidden_layer[0]))
		elif(len(hidden_layer)>1):
			for i in range(len(hidden_layer)-1):
				if i==0:
					self.hidden.append(nn.Linear(input_size, hidden_layer[i]))
					self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
				else:
					self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		self.reset_parameters(hidden_layer)
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], self.nClass)
		#pdb.set_trace()

	def reset_parameters(self,hidden_layer):
		#pdb.set_trace()
		hL = 0		
		while True:
			#pdb.set_trace()
			if self.hActFunc[hL].upper() in ['SIGMOID','TANH']:
				#pdb.set_trace()
				torch.nn.init.xavier_uniform_(self.hidden[hL].weight)
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
				#continue
			elif self.hActFunc[hL].upper() == 'RELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif self.hActFunc[hL].upper() == 'LRELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='leaky_relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			if hL == len(hidden_layer)-1:
				break
			hL += 1

	def forward(self, x):
		# Feedforward
		for l in range(len(self.hidden)):
			if self.hActFunc[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
		output= F.softmax(self.out(x), dim=1)
		return output
		
	def forwardPre(self, x):
		# Feedforward
		for l in range(self.currentHL+1):
			if self.hActFunc[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.hActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
		output= F.softmax(self.out(x), dim=1)
		return output

	def setHiddenWeight(self,W,b):
		for i in range(len(self.hidden)):
			self.hidden[i].bias.data=b[i].float()
			self.hidden[i].weight.data=W[i].float()

	def setOutputWeight(self,W,b):
		self.out.bias.data=b.float()
		self.out.weight.data=W.float()
		
	def splitData(self,labeled_data,split_ratio):
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
	
		return train_set,test_set
		
	def preTrain(self,dataLoader,verbose):

		#loop to do layer-wise pre-training
		for d in range(len(self.hLayer)):

			#pick one hidden layer one at a time for pre-training
			hidden_layer = self.hLayer[:d+1]
			self.currentHL = d

			if verbose:
				if d==0:
					print('Pre-training layer',self.inputDim,'-->[',self.hLayer[d],']-->',self.nClass)
				else:					
					print('Pre-training layer',self.hLayer[d-1],'-->[',hidden_layer[d],']-->',self.nClass)

			#initialize the network weight and bias
			self.initNet(self.inputDim,hidden_layer)

			# reset weights and biases by pretrained layers.
			if d>0:
				for l in range(d):
					# initialize the net					
					self.hidden[l].weight.data = deepcopy(preW[l])
					self.hidden[l].bias.data = deepcopy(preB[l])
					self.hidden[l].weight.requires_grad=False
					self.hidden[l].bias.requires_grad=False

			# set loss function
			criterion = nn.CrossEntropyLoss()

			# set optimization function
			if self.params['optimizationFunc'].upper()=='ADAM':
				optimizer = torch.optim.Adam(self.parameters(),lr=self.params['learningRate'],amsgrad=True)
			elif self.params['optimizationFunc'].upper()=='SGD':
				optimizer = torch.optim.SGD(self.parameters(),lr=self.params['learningRate'],momentum=self.params['momentum'])

			# Load the model to device
			self.to(self.device)
			numEpochs = self.params['noItrPre']
			# Start training			
			for epoch in range(numEpochs):
				error=[]
				for i, (trInput, trOutput) in enumerate(dataLoader):  
					# Move tensors to the configured device
					trInput = trInput.to(self.device)
					trOutput = trOutput.to(self.device)

					# Forward pass
					outputs = self.forwardPre(trInput)
					loss = criterion(outputs, trOutput)
					
					# Check for regularization
					if self.l1Penalty != 0 or self.l2Penalty != 0:
						l1RegLoss,l2RegLoss = torch.tensor([0.0],requires_grad=True).to(self.device), torch.tensor([0.0],requires_grad=True).to(self.device)
						if self.l1Penalty != 0 and self.l2Penalty == 0:
							for W in self.parameters():
								l1RegLoss += W.norm(1)
							loss = loss + self.l1Penalty * l1RegLoss
						elif self.l1Penalty == 0 and self.l2Penalty != 0:
							for W in self.parameters():
								l2RegLoss += W.norm(2)**2
							loss = loss + 0.5 * self.l2Penalty * l2RegLoss
						elif self.l1Penalty != 0 and self.l2Penalty != 0:
							for W in self.parameters():
								l2RegLoss += W.norm(2)**2
								l1RegLoss += W.norm(1)
							loss = loss + self.l1Penalty * l1RegLoss + 0.5 * self.l2Penalty * l2RegLoss
					
					error.append(loss.item())

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				self.epochError.append(np.mean(error))
				if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
					print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs, self.epochError[-1]))
			
			#variable to store pre-trained weight and bias
			if d <len(self.hLayer)-1:
				preW=[]
				preB=[]
				for h in range(len(hidden_layer)):
					preW.append(self.hidden[h].weight)
					preB.append(self.hidden[h].bias)
				#preW.append(self.out.weight)
				#preB.append(self.out.bias)

		#now set requires_grad =True for all the layers
		for l in range(len(hidden_layer)):			
			self.hidden[l].weight.requires_grad=True			
			self.hidden[l].bias.requires_grad=True
			
		self.out.weight.requires_grad=True
		self.out.bias.requires_grad=True

		if verbose:
			print('Pre-training is done.')
		
	def train(self,dataLoader,verbose):

		criterion = nn.CrossEntropyLoss()

		# set optimization function
		if self.params['optimizationFunc'].upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(),lr=self.params['learningRate'],amsgrad=True)
		elif self.params['optimizationFunc'].upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(),lr=self.params['learningRate'],momentum=self.params['momentum'])

		# Load the model to device
		self.to(self.device)
		
		numEpochs = self.params['noItrPost']
		for epoch in range(numEpochs):
			error=[]
			for i, (sample, labels) in enumerate(dataLoader):  
				# Move tensors to the configured device
				sample = sample.to(self.device)
				labels = labels.to(self.device)

				# Forward pass
				outputs = self.forward(sample)
				loss = criterion(outputs, labels)
				error.append(loss.item())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			self.epochError.append(np.mean(error))
			if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs,self.epochError[-1]))

	def trainWithEarlyStop(self,dataLoader,valData,valLabels,verbose):

		criterion = nn.CrossEntropyLoss()

		# set optimization function
		if self.params['optimizationFunc'].upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(),lr=self.params['learningRate'],amsgrad=True)
		elif self.params['optimizationFunc'].upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(),lr=self.params['learningRate'],momentum=self.params['momentum'])

		# Load the model to device
		self.to(self.device)

		#load validation data in GPU
		if self.earlyStop:
			valData = valData.to(self.device)
			valLabels = valLabels.to(self.device)

		epoch = 0
		trainingDone = False
		valError = []
		bestIndex = 0
		tmpW = deque(maxlen = self.earlyStopWindow)
		tmpB = deque(maxlen = self.earlyStopWindow)
		tmpOut = deque(maxlen = self.earlyStopWindow)
		while not(trainingDone):

			error = []			
			for i, (trInput, trOutput) in enumerate(dataLoader):  
				# Move tensors to the configured device
				trInput = trInput.to(self.device)
				trOutput = trOutput.to(self.device)

				# Forward pass
				outputs = self.forward(trInput)
				loss = criterion(outputs, trOutput)				
				error.append(loss.item())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			#calculate validation error
			with torch.no_grad():
				valOutput = self.forward(valData)
				valLoss = criterion(valOutput, valLabels)
			valError.append(valLoss.item())

			#increment epoch
			epoch += 1
			
			#store the weights and bias in temporary variables. The best parameters will be picked up from this later.
			tmpW.append([deepcopy(self.hidden[i].weight.data) for i in range(len(self.hidden))])
			tmpB.append([deepcopy(self.hidden[i].bias.data) for i in range(len(self.hidden))])
			tmpOut.append([deepcopy(self.out.weight.data),deepcopy(self.out.bias.data)])

			# now check for conditions for to stop training
			# check if validation error is increasing
			
			if epoch >= self.earlyStopWindow:
				#pdb.set_trace()
				errors = valError[-self.earlyStopWindow:]
				e1,e2 = np.mean(errors[:int(self.earlyStopWindow/2)]),np.mean(errors[int(self.earlyStopWindow/2):])
				if e1 <= e2: # validation error is increasing so stop training
					trainingDone = True
					bestIndex = np.where(errors == np.min(errors))[0][0]
					#print('Finetuning:Exiting from e1<=e2 check')

				elif e1-e2 <= self.errorCutoff: #validation error is not decreasing too much, so stop training
					trainingDone = True
					bestIndex = np.where(errors == np.min(errors))[0][0]
					#print('Finetuning:Exiting from e1-e2 < cutoff check')

			self.epochError.append(np.mean(error))
			if verbose and (epoch % 10) == 0:
			#if (epoch % 10) == 0:
				print ('Epoch {}, Trainingg Loss: {:.6f} Validation Loss: {:.6f}'.format(epoch, self.epochError[-1],valError[-1]))
				
		#store the weight and bias of hidden layer
		for i in range(len(self.hidden)):
			self.hidden[i].weight.data = deepcopy(tmpW[bestIndex][i])
			self.hidden[i].bias.data = deepcopy(tmpB[bestIndex][i])
		self.out.weight.data = deepcopy(tmpOut[bestIndex][0])
		self.out.bias.data = deepcopy(tmpOut[bestIndex][1])


	def _fit(self,trData,trLabels,preTraining=True,cudaDeviceId=0,verbose=False):
		
		#import custome modules:
		from calcom.classifiers._centroidencoder.utilityDBN import standardizeData
		
		self.inputDim,self.nClass = np.shape(trData)[1],len(np.unique(trLabels))
		
		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))
		
		#check for early stopping. If the flag is true than keep aside some portion of training data for validation
		if self.earlyStop:
			trLabels = np.array(trLabels).reshape(-1,1)
			lTrData,lValData = self.splitData(np.hstack((trData,trLabels)),split_ratio=0.8)
			trData,trLabels = lTrData[:,:-1],lTrData[:,-1].astype(int)
			valData,valLabels = lValData[:,:-1],lValData[:,-1]
			valData = torch.from_numpy(valData).float()
			valLabels = torch.from_numpy(valLabels.astype(int))
		
		if self.params['standardizeFlag']:
		#standardize data
			mu,sd,trData = standardizeData(trData)
			self.trMu = mu
			self.trSd = sd
		
		#Prepare data for torch
		trLabels = np.array(trLabels) # convert CCList() to np.array()
		#trDataTorch = Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(trLabels.flatten().astype(int)))
		trDataTorch = Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(trLabels))
		dataLoader = Data.DataLoader(dataset=trDataTorch,batch_size=self.params['miniBatchSize'],shuffle=True)
		
		if preTraining:
			self.preTrain(dataLoader,verbose)
		else:
			self.currentHL = self.nLayers
			#initialize the network weight and bias
			self.initNet(self.inputDim,self.hLayer)
		
		if self.earlyStop:			
			self.trainWithEarlyStop(dataLoader,valData,valLabels,verbose)
		else:			
			self.train(dataLoader,verbose)

	def _predict(self,x):
		
		#import custom module
		from calcom.classifiers._centroidencoder.utilityDBN import standardizeData
		from calcom.io import CCList

		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = standardizeData(x,self.trMu,self.trSd)
		x = torch.from_numpy(x).float().to(self.device)

		with torch.no_grad():
			fOut = self.forward(x)
		fOut = fOut.to('cpu').numpy()
		predictedLabels = (np.argmax(fOut,axis=1))
		return CCList(predictedLabels)
		
	def decision_function(self,x):
		#import custom module
		from calcom.classifiers._centroidencoder.utilityDBN import standardizeData
		from calcom.io import CCList

		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = standardizeData(x,self.trMu,self.trSd)
		x = torch.from_numpy(x).float().to(self.device)
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for l in range(len(self.hLayerPre)):
				if self.hActFunc[l].upper()=='SIGMOID':
					x = torch.sigmoid(self.classifier.hidden[l](x))
				elif self.hActFunc[l].upper()=='TANH':
					x = torch.tanh(self.classifier.hidden[l](x))
				elif self.hActFunc[l].upper()=='RELU':
					x = torch.relu(self.classifier.hidden[l](x))
				elif self.hActFunc[l].upper()=='LRELU':
					x = F.leaky_relu(self.classifier.hidden[l](x),inplace=False)
			fOut = F.softmax(self.classifier.out(x), dim=1)
			fOut = fOut.to('cpu').numpy()
		return fOut

	def visualize(self,*args):
		pass
