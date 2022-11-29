import pdb
import numpy as np
from copy import deepcopy

import torch

import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class NeuralNet(nn.Module):
	def __init__(self,input_size,hidden_layer,num_classes,hiddenActivation='relu',earlyStop=False):
		#hiddenActivation: this can be a list if you want to apply different activation at different layer.
		super(NeuralNet, self).__init__()
		self.hidden = nn.ModuleList()
		if isinstance(hiddenActivation,str):
			self.hiddenActivation = [hiddenActivation for i in range(len(hidden_layer))]
		elif isinstance(hiddenActivation,list):
			self.hiddenActivation = hiddenActivation
		self.epochError=[]
		self.device = None
		self.earlyStop = False
		self.earlyStopWindow,self.errorCutoff = None,None
		if earlyStop:
			self.earlyStop,self.earlyStopWindow,self.errorCutoff = earlyStop, 50, 10**(-2)#early stopping window and cut-off.
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
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], num_classes)

	def forward(self, x):
		# Feedforward
		for l in range(len(self.hidden)):
			if self.hiddenActivation[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hiddenActivation[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hiddenActivation[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.hiddenActivation[l].upper()=='LRELU':
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
		
	def train(self,dataLoader,optimizationFunc,learningRate,m,numEpochs,verbose):

		criterion = nn.CrossEntropyLoss()
		if optimizationFunc.upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(), lr=learningRate,amsgrad=True)
		elif optimizationFunc.upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(), lr=learningRate,momentum=m)

		self.to(self.device)
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

	def trainWithEarlyStop(self,dataLoader,valData,valLabels,optimizationFunc,learningRate,m,verbose):

		criterion = nn.CrossEntropyLoss()		
		# set optimization function
		if optimizationFunc.upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)
		elif optimizationFunc.upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(),lr=learningRate,momentum=m)

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

	
	def fit(self,dataLoader,valData,valLabels,optimizationFunc='Adam',learningRate=0.001,m=0,numEpochs=100,cudaDeviceId=0,verbose=False):
		
		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))
		
		if self.earlyStop:			
			self.trainWithEarlyStop(dataLoader,valData,valLabels,optimizationFunc,learningRate,m,verbose)
		else:
			self.train(dataLoader,optimizationFunc,learningRate,m,numEpochs,verbose)

	def predict(self,x):

		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = standardizeData(x,self.trMu,self.trSd)
		x = torch.from_numpy(x).float().to(self.device)

		with torch.no_grad():
			fOut = self.forward(x)
		fOut = fOut.to('cpu').numpy()
		predictedLabels = (np.argmax(fOut,axis=1)).reshape(-1,1)
		return predictedLabels
