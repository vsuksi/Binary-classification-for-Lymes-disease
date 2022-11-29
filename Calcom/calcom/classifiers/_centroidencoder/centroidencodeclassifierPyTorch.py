import pdb
from calcom.classifiers._abstractclassifier import AbstractClassifier

import numpy as np
import math
from copy import deepcopy

# Need to handle torch as an optional package.
# Quick fix: try/except the entire class; if we can't import 
# any of these packages. then replace with a dummy class.


# Watch these carefully for errors; __init__ for classifiers 
# should check for the torch dependency properly but who knows.
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable



class CentroidencoderClassifierPyTorch(nn.Module,AbstractClassifier):
	def __init__(self, netConfig={}):
		super(CentroidencoderClassifierPyTorch, self).__init__()

		#first do the default set up of parameters in a dictionary
		self.params = {}
		self.params['hLayer'] = [50,25,50]
		self.params['hActFunc'] = ['lrelu','lrelu','lrelu']
		self.params['earlyStop'] = True
		self.params['earlyStopWindow'] = 50
		self.params['errorCutoff'] = 10**(-2)
		self.params['noItrPre'] = 10
		self.params['noItrPost'] = 40
		self.params['noItrSoftmax'] = 10
		self.params['noItrFinetune'] = 10
		self.params['l1Penalty'], self.params['l2Penalty'] = 0.0,0.0
		self.params['linearDecoder'] = False
		self.params['optimizationFunc'] = 'Adam'
		self.params['learningRate'] = 0.001
		self.params['miniBatchSize'] = 32
		self.params['momentum'] = 0.8 # this will be used only for SGD. Adam doesn't need this.
		self.params['standardizeFlag'] = True

		#now check if the dictionary is passed.
		if len(netConfig.keys()) != 0:			
			self.params['hLayer'],self.params['hLayerPost'] = deepcopy(netConfig['hL']),deepcopy(netConfig['hL'])
			for i in range(len(self.params['hLayerPost'])-1,0,-1):
				self.params['hLayerPost'].extend([self.params['hLayerPost'][i-1]])
			if 'earlyStop' in netConfig.keys():
				self.params['earlyStop'] = netConfig['earlyStop']
				self.params['earlyStopWindow'] = 50
				self.params['errorCutoff'] = 10**(-2)
			if 'optimizationFunc' in netConfig.keys(): self.params['optimizationFunc'] = netConfig['optimizationFunc']
			if 'learningRate' in netConfig.keys(): self.params['learningRate'] = netConfig['learningRate']
			if 'miniBatchSize' in netConfig.keys(): self.params['miniBatchSize'] = netConfig['miniBatchSize']
			if 'standardizeFlag' in netConfig.keys(): self.params['standardizeFlag'] = netConfig['standardizeFlag']
			if 'l1Penalty' in netConfig.keys(): self.params['l1Penalty'] = netConfig['l1Penalty']
			if 'l2Penalty' in netConfig.keys(): self.params['l2Penalty'] = netConfig['l2Penalty']
			if 'linearDecoder' in netConfig.keys(): params['linearDecoder'] = netConfig['linearDecoder']


		#internal variables
		self.inputDim,self.outputDim = None,None
		self.epochError = []
		self.trMu = []
		self.trSd = []
		self.tmpPreHActFunc = []
		self.preTrW = []
		self.preTrB = []
		self.device = None
		self.nClass = None
		self.classifier = None
		self.fineTuneW = []
		self.fineTuneB = []
		self.nLayers = len(self.params['hLayer'])
		self.hLayerPre = self.params['hLayer'][:int(np.ceil(self.nLayers/2))]
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
		if 'noItrSoftmax' in params.keys(): self.params['noItrSoftmax'] = params.pop('noItrSoftmax')
		if 'noItrFinetune' in params.keys(): self.params['noItrFinetune'] = params.pop('noItrFinetune')
		if 'l1Penalty' in params.keys(): self.params['l1Penalty'] = params.pop('l1Penalty')
		if 'l2Penalty' in params.keys(): self.params['l2Penalty'] = params.pop('l2Penalty')
		
		self.nLayers = len(self.params['hLayer'])
		self.hLayerPre = self.params['hLayer'][:int(np.ceil(self.nLayers/2))]
		self.hLayer = self.params['hLayer']
		self.hActFunc = self.params['hActFunc']
		self.l1Penalty,self.l2Penalty = self.params['l1Penalty'],self.params['l2Penalty']
		self.earlyStop = self.params['earlyStop']
		self.earlyStopWindow, self.errorCutoff= self.params['earlyStopWindow'],self.params['errorCutoff']
		

	def initNet(self,input_size,hidden_layer):
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
		self.out = nn.Linear(hidden_layer[-1], input_size)

	def reset_parameters(self,hidden_layer):
		#pdb.set_trace()
		tmpActFunc = self.params['hActFunc'][:int(np.ceil(len(hidden_layer)/2))]
		for i in range(len(tmpActFunc)-1,0,-1):
			tmpActFunc.extend([tmpActFunc[i-1]])
		hL = 0
		
		while True:
			#pdb.set_trace()
			if tmpActFunc[hL].upper() in ['SIGMOID','TANH']:
				#pdb.set_trace()
				torch.nn.init.xavier_uniform_(self.hidden[hL].weight)
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
				#continue
			elif tmpActFunc[hL].upper() == 'RELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif tmpActFunc[hL].upper() == 'LRELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='leaky_relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			if hL == len(hidden_layer)-1:
				break
			hL += 1

	def forwardPre(self, x):
		# Feedforward
		for l in range(len(self.hidden)):
			if self.tmpPreHActFunc[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear
				x = self.hidden[l](x)

		return self.out(x)
	
	def forwardPost(self, x):
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
			else:#default is linear				
				x = self.hidden[l](x)

		return self.out(x)

	def softmaxData(self,x):
		# run forward pass
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for l in range(len(self.hLayerPre)):
				if self.hActFunc[l].upper()=='SIGMOID':
					x = torch.sigmoid(self.hidden[l](x))
				elif self.hActFunc[l].upper()=='TANH':
					x = torch.tanh(self.hidden[l](x))
				elif self.hActFunc[l].upper()=='RELU':
					x = torch.relu(self.hidden[l](x))
				elif self.hActFunc[l].upper()=='LRELU':
					x = F.leaky_relu(self.hidden[l](x),inplace=False)
				else:#default is linear
					x = self.hidden[l](x)
		return x

	def returnTransformedData(self,x):
		fOut=[x]
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for layer in self.hidden:
				fOut.append(self.hiddenActivation(layer(fOut[-1])))
			if self.output_activation.upper()=='SIGMOID':
				fOut.append(torch.sigmoid(self.out(fOut[-1])))
			else:
				fOut.append(self.out(fOut[-1]))
		return fOut[1:]#Ignore the original input

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
		for d in range(len(self.hLayerPre)):
			
			#set the hidden layer structure for a bottleneck architecture
			hidden_layer=self.hLayer[:d+1]
			self.tmpPreHActFunc=self.hActFunc[:d+1]
			for i in range(len(hidden_layer)-1,0,-1):
				hidden_layer.extend([hidden_layer[i-1]])
				self.tmpPreHActFunc.extend([self.tmpPreHActFunc[i-1]])

			if verbose:
				if d==0:
					print('Pre-training layer [',self.inputDim,'-->',hidden_layer[0],'-->',self.inputDim,']')
				else:
					index=int(len(hidden_layer)/2)
					print('Pre-training layer [',hidden_layer[index-1],'-->',hidden_layer[index],'-->',hidden_layer[index+1],']')			

			#initialize the network weight and bias
			self.initNet(self.inputDim,hidden_layer)

			#freeze pretrained layers
			if d>0:
				j=0#index for preW and preB
				for l in range(len(hidden_layer)):
					if (l==d) or (l==(d+1)):
						continue
					else:
						self.hidden[l].weight=preW[j]
						self.hidden[l].weight.requires_grad=False
						self.hidden[l].bias=preB[j]
						self.hidden[l].bias.requires_grad=False
						j+=1
				self.out.weight=preW[-1]
				self.out.weight.requires_grad=False
				self.out.bias=preB[-1]
				self.out.bias.requires_grad=False

			# set loss function
			criterion = nn.MSELoss()

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
				preW.append(self.out.weight)
				preB.append(self.out.bias)

		#now set requires_grad =True for all the layers
		for l in range(len(hidden_layer)):			
			self.hidden[l].weight.requires_grad=True			
			self.hidden[l].bias.requires_grad=True
			
		self.out.weight.requires_grad=True
		self.out.bias.requires_grad=True
		
		if verbose:
			print('Pre-training is done.')


	def postTrain(self,dataLoader,verbose):

		criterion = nn.MSELoss()
		
		# set optimization function
		if self.params['optimizationFunc'].upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(),lr=self.params['learningRate'],amsgrad=True)
		elif self.params['optimizationFunc'].upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(),lr=self.params['learningRate'],momentum=self.params['momentum'])

		# Load the model to device
		self.to(self.device)
		
		# Start training
		if verbose:
			print('Training network:',self.inputDim,'-->',self.hLayer,'-->',self.inputDim)
		
		numEpochs = self.params['noItrPost']
		for epoch in range(numEpochs):
			error = []
			valError = []
			for i, (trInput, trOutput) in enumerate(dataLoader):  
				# Move tensors to the configured device
				trInput = trInput.to(self.device)
				trOutput = trOutput.to(self.device)

				# Forward pass
				outputs = self.forwardPost(trInput)
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

		#store the weight and bias of hidden layer
		self.preTrW=[deepcopy(self.hidden[i].weight.data) for i in range(len(self.hLayerPre))]
		self.preTrB=[deepcopy(self.hidden[i].bias.data) for i in range(len(self.hLayerPre))]

	def postTrainEarlyStop(self,dataLoader,valData,valTarget,verbose):

		criterion = nn.MSELoss()
		
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
			valTarget = valTarget.to(self.device)
		
		# Start training
		if verbose:
			print('Training network:',self.inputDim,'-->',self.hLayer,'-->',self.inputDim)
		
		epoch = 0
		trainingDone = False
		valError = []
		bestIndex = 0
		tmpW = deque(maxlen = self.earlyStopWindow)
		tmpB = deque(maxlen = self.earlyStopWindow)
		while not(trainingDone):
			error = []			
			for i, (trInput, trOutput) in enumerate(dataLoader):  
				# Move tensors to the configured device
				trInput = trInput.to(self.device)
				trOutput = trOutput.to(self.device)

				# Forward pass
				outputs = self.forwardPost(trInput)
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

			#calculate validation error			
			valOutput = self.forwardPost(valData)
			valLoss = criterion(valOutput, valTarget)
			valError.append(valLoss.item())

			#increment epoch
			epoch += 1

			#store the weights and bias in temporary variables. The best parameters will be picked up from this later.
			tmpW.append([deepcopy(self.hidden[i].weight.data) for i in range(len(self.hLayer))])
			tmpB.append([deepcopy(self.hidden[i].bias.data) for i in range(len(self.hLayer))])

			# now check for conditions for to stop training
			# check if validation error is increasing
			
			if epoch >= self.earlyStopWindow:
				#pdb.set_trace()
				errors = valError[-self.earlyStopWindow:]
				e1,e2 = np.mean(errors[:int(self.earlyStopWindow/2)]),np.mean(errors[int(self.earlyStopWindow/2):])
				if e1 <= e2: # validation error is increasing so stop training
					trainingDone = True
					bestIndex = np.where(errors == np.min(errors))[0][0]
					#print('postTrainEarlyStop:Exiting from e1<=e2 check')
				elif e1-e2 <= self.errorCutoff: #validation error is not decreasing too much, so stop training
					trainingDone = True
					bestIndex = np.where(errors == np.min(errors))[0][0]
					#print('postTrainEarlyStop:Exiting from e1-e2 < cutoff check')

			self.epochError.append(np.mean(error))
				
			if verbose and (epoch % 10) == 0:
			#if (epoch % 10) == 0:
				print ('Epoch {}, Trainingg Loss: {:.6f} Validation Loss: {:.6f}'.format(epoch, self.epochError[-1],valError[-1]))

		#store the weight and bias of hidden layer
		self.preTrW = [deepcopy(tmpW[bestIndex][i]) for i in range(len(self.hLayerPre))]
		self.preTrB = [deepcopy(tmpB[bestIndex][i]) for i in range(len(self.hLayerPre))]
		#pdb.set_trace()

	def trainSoftmax(self,trData,trLabels,cudaDeviceId,verbose):
		
		#import custome modules:
		from calcom.classifiers._centroidencoder import softmaxClassifierPyTorch as softmax

		#define a softmax layer with the bottleneck output
		softmaxModel = softmax.Softmax(self.preTrW[-1].shape[0],self.nClass)
		softMaxData = self.softmaxData(torch.from_numpy(trData).float().to(self.device))
		
		#prepare data for softmax
		trLabels = np.array(trLabels) # convert CCList() to np.array()
		softmaxTrDataTorch = Data.TensorDataset(softMaxData,torch.from_numpy(trLabels))
		softmaxTrDataLoader = Data.DataLoader(dataset=softmaxTrDataTorch,batch_size=self.params['miniBatchSize'],shuffle=True)
		if verbose:
			print('Training softmax layer')
		softmaxModel.fit(softmaxTrDataLoader,optimizationFunc=self.params['optimizationFunc'],learningRate=self.params['learningRate'],
			numEpochs=self.params['noItrSoftmax'],cudaDeviceId=cudaDeviceId,verbose=verbose)
		self.preTrW.append(deepcopy(softmaxModel.softmaxLayer.weight.data))
		self.preTrB.append(deepcopy(softmaxModel.softmaxLayer.bias.data))

	def runFinetuning(self,dataLoader,valData,valLabels,cudaDeviceId,verbose):

		#import custome modules:
		from calcom.classifiers._centroidencoder import deepANNClassifierPyTorch as dc
		
		# first define the network
		actFunc = self.hActFunc[:len(self.hLayerPre)]
		ann = dc.NeuralNet(self.inputDim, self.hLayerPre, self.nClass,actFunc,self.earlyStop)

		# now reset the pre-trained weights and biases
		for l in range(len(ann.hidden)):
			ann.hidden[l].weight.data = deepcopy(self.preTrW[l])
			ann.hidden[l].bias.data = deepcopy(self.preTrB[l])
		ann.out.weight.data = deepcopy(self.preTrW[-1])
		ann.out.bias.data = deepcopy(self.preTrB[-1])

		# now fine tune the classifier
		if verbose:
			print('Running fine tuning')
		ann.fit(dataLoader,valData,valLabels,optimizationFunc=self.params['optimizationFunc'],learningRate=self.params['learningRate'],
			numEpochs=self.params['noItrFinetune'],cudaDeviceId=cudaDeviceId,verbose=verbose)
		
		self.classifier = deepcopy(ann)


	def _fit(self,trData,trLabels,preTraining=True,cudaDeviceId=0,verbose=False):
		
		#import custome modules:
		from calcom.classifiers._centroidencoder.utilityDBN import createOutputAsCentroids,standardizeData

		self.inputDim,self.outputDim = np.shape(trData)[1],np.shape(trData)[1]
		#internal_labels = trLabels

		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))

		#check for early stopping. If the flag is true than keep aside some portion of training data for validation
		if self.earlyStop:
			#pdb.set_trace()
			trLabels = np.array(trLabels).reshape(-1,1)
			lTrData,lValData = self.splitData(np.hstack((trData,trLabels)),split_ratio=0.8)
			trData,trLabels = lTrData[:,:-1],lTrData[:,-1].astype(int)
			valData,valLabels = lValData[:,:-1],lValData[:,-1]
			valLabels = torch.from_numpy(valLabels.astype(int))

		if self.params['standardizeFlag']:
		#standardize data
			mu,sd,trData = standardizeData(trData)
			self.trMu = mu
			self.trSd = sd
		
		self.nClass = len(np.unique(trLabels))
		
		#create target: centroid for each class
		target = createOutputAsCentroids(trData,trLabels)

		#Prepare data for torch
		trDataTorch = Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(target).float())
		dataLoader = Data.DataLoader(dataset=trDataTorch,batch_size=self.params['miniBatchSize'],shuffle=True)
		#pdb.set_trace()
		#layer-wise pre-training with CE cost
		if preTraining:
			self.preTrain(dataLoader,verbose)
		else:
			#initialize the network weight and bias
			self.initNet(self.inputDim,self.hLayer)
		
		#post training CE cost with early stopping
		if self.params['earlyStop']:
			#pdb.set_trace()
			valData = standardizeData(valData,mu,sd)
			valTarget = createOutputAsCentroids(valData,valLabels)
			valData,valTarget = torch.from_numpy(valData).float(),torch.from_numpy(valTarget).float()

			self.postTrainEarlyStop(dataLoader,valData,valTarget,verbose)
		else:
			self.postTrain(dataLoader,verbose)
		
		# train a softmax layer
		self.trainSoftmax(trData,trLabels,cudaDeviceId,verbose)

		# prepare data for finetuning
		trLabels = np.array(trLabels) # convert CCList() to np.array()
		#fineTuningData = Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(trLabels.flatten().astype(int)))
		fineTuningData = Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(trLabels))
		fineTuningLoader = Data.DataLoader(dataset=fineTuningData,batch_size=self.params['miniBatchSize'],shuffle=True)

		if self.earlyStop:
			self.runFinetuning(fineTuningLoader,valData,valLabels,cudaDeviceId,verbose)
		else:
			self.runFinetuning(fineTuningLoader,[],[],cudaDeviceId,verbose)

		
	def _predict(self,x):

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
		#pdb.set_trace()
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
		
#
