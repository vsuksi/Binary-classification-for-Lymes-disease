import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#if not have_dependencies:
#    raise Exception('Import of torch dependencies failed; this class is not supported.')

class Softmax(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Softmax, self).__init__()
		self.softmaxLayer = nn.Linear(input_size, num_classes)
		self.epochError=[]
		self.device = None
		
	def forward(self,x):
		output= F.softmax(self.softmaxLayer(x), dim=1)
		return output

	def fit(self,dataLoader,optimizationFunc='Adam',learningRate=0.001,m=0.8,numEpochs=10,cudaDeviceId=0,verbose=False):
		
		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))

		criterion = nn.CrossEntropyLoss()
		
		if optimizationFunc.upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
		elif optimizationFunc.upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(), lr=learningRate,momentum=m)
		total_step = len(dataLoader)
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
				print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs, self.epochError[-1]))

	def predict(self,dataLoader,device=''):

		with torch.no_grad():
			correct = 0
			total = 0
			pVals=[]
			pLabels=[]
			for sample, labels in dataLoader:
				sample = sample.to(self.device)
				labels = labels.to(self.device)
				outputs = self.forward(sample)
				_, predicted = torch.max(outputs.data, 1)
				pLabels.append(predicted)
				pVals.append(outputs)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		pVals=np.vstack((pVals))
		pLabels=np.hstack((pLabels))
		print('Accuracy of the network : {} %'.format(100 * correct / total))
		return pVals,pLabels
		
