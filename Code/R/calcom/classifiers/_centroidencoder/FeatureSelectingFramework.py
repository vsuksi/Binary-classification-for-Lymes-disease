'''
Owner's Name:Tomojit Ghosh
Program Name: FeatureSelectingFramewrowk.py
Purpose: For my PhD thesis. Feature selection using a sparsity promoting layer in between input layer and first hidden layer. I'm trying 
		to make this code as general as possible so that it can used in conjunction with autoencoder, centrod-encoder and ANN classifier
'''
#import pdb
import numpy as np
#import sys
from copy import copy
from calcom.classifiers._centroidencoder import nonLinearOptimizationAlgorithms as scg
from scipy.special import expit
#from utilityDBN import MSE
#from conjugateGradientSearch import CGS
import matplotlib.pyplot as plt
from copy import copy

class FeatureSelectingFramework:
	def __init__(self,netConfig):
		self.inputDim=netConfig['inputL']
		self.outputDim=netConfig['outputL']
		self.hLayer=copy(netConfig['hL'])
		self.nUnits=[self.inputDim]+list(self.hLayer)+[self.outputDim]
		self.nLayers=len(self.nUnits)-1
		self.netW=[]
		self.wLayout=[]
		self.Xmeans=None
		self.Xstds=None
		self.Tmeans=None
		self.Tstds=None
		self.errorFunc=netConfig['errorFunc'] 
		self.trained=False
		self.trErrorTrace=None
		self.valErrorTrace=None
		self.iteration=0
		self.layer_iteration=[]
		self.hlNo=0
		self.layer_error_trace=[]
		self.layer_weight_trace=[]
		self.actFunc=netConfig['actFunc']#This will contain the activation functions of all the hidden layers of parallel layers
		self.lActFunc=''
		self.nItr=netConfig['nItr']
		self.freezeLayerFlag=''
		self.postTrErr=[]
		self.preTrainingDone=False
		self.postTrainingDone=False
		self.runningPostTraining=False
		self.splW=[]
		self.l1Penalty=None
		self.l2Penalty=None
		self.WeightElimination=None #It's a list if passed to the program. 0th element is lambda and 1th element is w_0
		self.applyPenalty=False
		self.splWBeforePenalty=[]
		self.iterationWeights=[]
		self.usePreTrainedW=False

	def initWeight(self,nUnits):
		nLayers=len(nUnits)-1
		#W=[np.random.uniform(-np.sqrt(3),np.sqrt(3), size=(1+nUnits[i],nUnits[i+1])) / np.sqrt(nUnits[i]) for i in range(nLayers)]
		W=[np.random.uniform(-1,1, size=(1+nUnits[i],nUnits[i+1])) / np.sqrt(nUnits[i]) for i in range(nLayers)]
		if self.netW==[]:
			self.netW=[w for w in W]
		else:
			tmpW=[]
			self.netW[-1] = copy(W[0])
			self.netW.append(W[1])
		return W

	def setPreTrainedWeight(self,Ws):
		self.splW = Ws[0]
		self.netW=Ws[1:]
		self.usePreTrainedW=True

	def standardizeX(self,X):
		result=(X-self.Xmeans)/self.XstdsFixed
		result[:,self.Xconstant]=0.0
		return result

	def unstandardizeX(self,X):
		return self.Xstds*X + self.Xmeans

	def standardizeT(self,T):
		result=(T-self.Tmeans)/self.TstdsFixed
		result[:,self.Tconstant]=0.0
		return result

	def unstandardizeT(self,T):
		return self.Tstds*T+self.Tmeans

	def flattenD(self,Ws):
		return(np.hstack([W.flat for W in Ws]))

	def unFlattenD(self,Ws):
		#pdb.set_trace()
		sIndex=0
		tmpWs=[]
		Ws=Ws.reshape(1,-1)
		if self.freezeLayerFlag == True:
			for i in range(self.hlNo-1,self.hlNo+1,1):
				d1=np.shape(self.netW[i])[0]
				d2=np.shape(self.netW[i])[1]
				self.netW[i]=(Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2)
				sIndex=sIndex+(d1*d2)
		else:
			for i in range(len(self.netW)):
				d1=np.shape(self.netW[i])[0]
				d2=np.shape(self.netW[i])[1]
				self.netW[i]=(Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2)
				sIndex=sIndex+(d1*d2)
 
	def feval(self,fName,*args):
		return eval(fName)(*args)
	
	def tanh(self,X):
		return np.tanh(X)

	def sigmoid(self,X):
		'''
		#"Numerically-stable sigmoid function."
		#Taken from http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
		Y=copy(X)
		Y[Y>=0]=1.0/(1+np.exp(-Y[Y>=0]))
		Y[Y<0]=np.exp(Y[Y<0])/(1+np.exp(Y[Y<0]))
		return Y
		'''
		return expit(X) #using the sigmoidal of scipy package

	def softplus(self,X):
		return np.log(1+np.exp(X))

	def rect(self,X):
		X[X<0]=0*X[X<0]
		return X

	def rectl(self,X):
		X[X<0]=0.01*X[X<0]
		return X

	def linear(self,X):
		return X

	def getTrErrorTrace(self):
		return self.trErrorTrace

	def getValErrorTrace(self):
		return self.valErrorTrace

	def forwardPass(self,D):
		#if self.__class__.__name__=='AE':
			#pdb.set_trace()
		#This function will return the network output for each layer.'key' is the identifier for each layer
		#print ('Class Name: ',self.__class__.__name__)
		lOut=[D]
		lLength=len(self.netW)
		#pdb.set_trace()
		for j in range(lLength):
			d=np.dot(lOut[-1],self.netW[j][1:,:])+self.netW[j][0]#first row in the weight is the bias
			#Take the activation function from the dictionary and apply it
			lOut.append(self.feval('self.'+self.lActFunc,d) if j<lLength-1 else d)
		return lOut
		
	def forwardPassPost(self,D):
		#if self.__class__.__name__=='AE':
			#pdb.set_trace()
		#This function will return the network output for each layer.'key' is the identifier for each layer
		#print ('Class Name: ',self.__class__.__name__)
		lOut=[D]
		lLength=len(self.netW)
		#pdb.set_trace()
		for j in range(lLength):
			d=np.dot(lOut[-1],self.netW[j][1:,:])+self.netW[j][0]#first row in the weight is the bias
			#Take the activation function from the dictionary and apply it
			lOut.append(self.feval('self.'+self.lActFunc[j],d) if j<lLength-1 else d)
		return lOut	

	def backwardPassFreezeLayer(self,error,lO):
		#This will return the partial derivatives for all the layers.
		deltas=[error]
		#pdb.set_trace()
		for l in range(len(self.netW)-1,0,-1):
			if 'sigmoid' in self.lActFunc:
			#Activation function as f(x)=1/(1+exp(-x))
			#f'(x)=f(x)(1-f(x)) I'm doing (lO[i]*(1-lO[i]))
				delta=(lO[l]*(1-lO[l]))*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'tanh' in self.lActFunc:
			#Activation function: f(x)=(1-exp(-x))/(1+exp(-x))
			#f'(x)=(1-f(x)^2) I'm doing ((1-lO[i]^2))
				delta=(1-lO[l]**2)*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'softplus' in self.lActFunc:
			#Activation function: f(x)=log(1+exp(x))
			#f'(x)=1/(1+exp(-x)), so use the sigmoid
				delta=self.sigmoid(lO[l])*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			#Rectifier unit
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0
			elif 'rect' in self.lActFunc:
				derivatives = 1*np.array(lO[l]>=0).astype(int)
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rectl' in self.lActFunc:
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'linear' in self.lActFunc:
				delta=(np.dot(deltas[-1],self.netW[l][1:,:].T))
			else:
				print('Wrong activation function')
			deltas.append(delta)
		deltas.reverse()
		dWs=[]
		for l in range(self.hlNo-1,self.hlNo+1,1):
			#dWs.append(np.vstack((np.dot(lO[l].T,deltas[l]),deltas[l].sum(0))))
			dWs.append(np.vstack((deltas[l].sum(0),np.dot(lO[l].T, deltas[l]))))#The first row is the bias
		return dWs

	def backwardPass(self,error,lO):
		#This will return the partial derivatives for all the layers.
		deltas=[error]
		for l in range(len(self.netW)-1,0,-1):
			if 'sigmoid' in self.lActFunc:
			#Activation function as f(x)=1/(1+exp(-x))
			#f'(x)=f(x)(1-f(x)) I'm doing (lO[i]*(1-lO[i]))
				delta=(lO[l]*(1-lO[l]))*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'tanh' in self.lActFunc:
			#Activation function: f(x)=(1-exp(-x))/(1+exp(-x))
			#f'(x)=(1-f(x)^2) I'm doing ((1-lO[i]^2))
				delta=(1-lO[l]**2)*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'softplus' in self.lActFunc:
			#Activation function: f(x)=log(1+exp(x))
			#f'(x)=1/(1+exp(-x)), so use the sigmoid
				delta=self.sigmoid(lO[l])*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rect' in self.lActFunc:
			#Rectifier unit function
				derivatives = 1*np.array(lO[l]>=0).astype(int)
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rectl' in self.lActFunc:
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'linear' in self.lActFunc:
				delta=(np.dot(deltas[-1],self.netW[l][1:,:].T))
			else:
				print('Wrong activation function')
			deltas.append(delta)
		deltas.reverse()
		dWs=[]
		for l in range(len(self.netW)):
			dWs.append(np.vstack((deltas[l].sum(0),np.dot(lO[l].T, deltas[l]))))#The first row is the bias
		return dWs
		
	def backwardPassPost(self,error,lO):
		#This will return the partial derivatives for all the layers.
		deltas=[error]
		#pdb.set_trace()
		for l in range(len(self.netW)-1,0,-1):
			lActFunc=self.actFunc[l-1]
			if 'sigmoid' in self.lActFunc:
			#Activation function as f(x)=1/(1+exp(-x))
			#f'(x)=f(x)(1-f(x)) I'm doing (lO[i]*(1-lO[i]))
				delta=(lO[l]*(1-lO[l]))*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'tanh' in self.lActFunc:
			#Activation function: f(x)=(1-exp(-x))/(1+exp(-x))
			#f'(x)=(1-f(x)^2) I'm doing ((1-lO[i]^2))
				delta=(1-lO[l]**2)*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'softplus' in self.lActFunc:
			#Activation function: f(x)=log(1+exp(x))
			#f'(x)=1/(1+exp(-x)), so use the sigmoid
				delta=self.sigmoid(lO[l])*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rect' in self.lActFunc:
			#Rectifier unit function
				derivatives = 1*np.array(lO[l]>=0).astype(int)
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rectl' in self.lActFunc:
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'linear' in self.lActFunc:
				delta=(np.dot(deltas[-1],self.netW[l][1:,:].T))
			else:
				print('Wrong activation function')
			deltas.append(delta)
		deltas.reverse()
		dWs=[]
		for l in range(len(self.netW)):
			dWs.append(np.vstack((deltas[l].sum(0),np.dot(lO[l].T, deltas[l]))))#The first row is the bias
		return dWs

	def backwardPassSPL(self,error,lO,trData):
		#This will return the partial derivatives for all the layers.
		deltas=[error]
		#pdb.set_trace()
		for l in range(len(self.netW)-1,0,-1):
			lActFunc=self.actFunc[l-1]
			if 'sigmoid' in lActFunc:
			#Activation function as f(x)=1/(1+exp(-x))
			#f'(x)=f(x)(1-f(x)) I'm doing (lO[i]*(1-lO[i]))
				delta=(lO[l]*(1-lO[l]))*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'tanh' in lActFunc:
			#Activation function: f(x)=(1-exp(-x))/(1+exp(-x))
			#f'(x)=(1-f(x)^2) I'm doing ((1-lO[i]^2))
				delta=(1-lO[l]**2)*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'softplus' in lActFunc:
			#Activation function: f(x)=log(1+exp(x))
			#f'(x)=1/(1+exp(-x)), so use the sigmoid
				delta=self.sigmoid(lO[l])*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rect' in lActFunc:
			#Rectifier unit function
				derivatives = 1*np.array(lO[l]>=0).astype(int)
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rectl' in lActFunc:
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'linear' in lActFunc:
				delta=(np.dot(deltas[-1],self.netW[l][1:,:].T))
			else:
				print('Wrong activation function')
			deltas.append(delta)
		splDelta = np.dot(deltas[-1],self.netW[0][1:,:].T)		
		dWs = [(trData*splDelta).sum(0)]
		deltas.reverse()
		for l in range(len(self.netW)):
			dWs.append(np.vstack((deltas[l].sum(0),np.dot(lO[l].T, deltas[l]))))#The first row is the bias
		#pdb.set_trace()
		return dWs

	def regenD(self,X):		
		Zs = self.forwardPass(X)
		Zs[-1][:] = Zs[-1][:]
		return Zs

	def calcError(self,cOut,tOut):
		nSamples=np.shape(cOut)[0]
		if self.errorFunc=='CE':
			return (self.calcLogProb(cOut)-tOut)/(nSamples*self.outputDim)
			#return (self.calcLogProb(cOut)-tOut)/nSamples
		elif self.errorFunc=='MSE':
			return (cOut-tOut)/(nSamples*self.outputDim)
		else:
			print('Wrong error function')

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

	def costFuncMSE(self,W,iData,tOut):
		self.unFlattenD(W)
		if self.runningPostTraining == True:
			cOut = self.forwardPassPost(iData)
		else:
			cOut = self.forwardPass(iData)
		if self.l1Penalty != None and self.l2Penalty != None:
			return 0.5*np.mean((cOut[-1] - tOut)**2) + self.l1Penalty*np.sum(np.abs(self.splW)) + 0.5*self.l2Penalty*np.sum(self.splW**2)
		elif self.l1Penalty != None and self.l2Penalty == None:				
			return 0.5*np.mean((cOut[-1] - tOut)**2) + self.l1Penalty*np.sum(np.abs(self.splW))
		elif self.l1Penalty == None and self.l2Penalty != None:
			return 0.5*np.mean((cOut[-1] - tOut)**2) + 0.5*self.l2Penalty*np.sum(self.splW**2)
		elif self.WeightElimination != None:
			#pdb.set_trace()
			penaltyTerm = self.WeightElimination[0]*sum([(self.splW[l]**2/(self.splW[l]**2+self.WeightElimination[1]**2)) for l in range(len(self.splW))])
			penaltyTerm = penaltyTerm/(self.outputDim*np.shape(iData)[0])
			error = 0.5*np.mean((cOut[-1] - tOut)**2) + penaltyTerm
			#print('Penalty term:',penaltyTerm,' Error:',error)
			return error
		else:
			return 0.5 * np.mean((cOut[-1] - tOut)**2)
		#return 0.5*np.mean((cOut[-1] - tOut)**2)

	def costFuncCE(self,W,iData,tOut):
		#if self.runningPostTraining == True:
		#pdb.set_trace()
		self.unFlattenD(W)
		if self.runningPostTraining == True:
			lOut = self.forwardPassPost(iData)
		else:
			lOut = self.forwardPass(iData)
		cOut=self.calcLogProb(lOut[-1])
		cOut[cOut==0]=np.finfo(np.float64).tiny
		if self.l1Penalty != None and self.l2Penalty != None:
			return -np.mean(tOut*(np.log(cOut))) + self.l1Penalty*np.sum(np.abs(self.splW)) + 0.5*self.l2Penalty*np.sum(self.splW**2)
		elif self.l1Penalty != None and self.l2Penalty == None:
			#pdb.set_trace()
			return -np.mean(tOut*(np.log(cOut))) + self.l1Penalty*np.sum(np.abs(self.splW))
		elif self.l1Penalty == None and self.l2Penalty != None:
			return -np.mean(tOut*(np.log(cOut))) + self.l2Penalty*np.sum(self.splW**2)
		elif self.WeightElimination != None:
			#pdb.set_trace()
			penaltyTerm = self.WeightElimination[0]*sum([(self.splW[l]**2/(self.splW[l]**2+self.WeightElimination[1]**2)) for l in range(len(self.splW))])
			penaltyTerm = penaltyTerm/(self.outputDim*np.shape(iData)[0])
			error = -np.mean(tOut*(np.log(cOut))) + penaltyTerm
			#print('Penalty term:',penaltyTerm,' Error:',error)
			return error
			#return -np.mean(tOut*(np.log(cOut))) + self.WeightElimination[0]*sum([(self.splW[l]**2/(self.splW[l]**2+self.WeightElimination[1]**2)) for l in range(len(self.splW))])
		else:
			return -np.mean(tOut*(np.log(cOut)))
		#return -np.mean(tOut*(np.log(cOut)))

	def calcMSE(self,W,iData,tOut):
		#This function will return the RMSE on training data. The error is calculated per sample per output dimension
		self.unFlattenD(W)		
		#pdb.set_trace()
		if self.runningPostTraining == True:
			cOut = self.forwardPassPost(iData)
		else:
			cOut = self.forwardPass(iData)
		squaredRes = (cOut[-1] - tOut)**2
		#mse = np.mean(np.sum(squaredRes,1),0)
		#return mse
		rmse = np.sqrt(np.mean(squaredRes))
		return rmse
		
	def calcCE(self,W,iData,tOut):
		#This function will return calculated per sample per output dimension
		#pdb.set_trace()
		self.unFlattenD(W)
		nSamples=np.shape(iData)[0]
		if self.runningPostTraining == True:
			lOut = self.forwardPassPost(iData)
		else:
			lOut = self.forwardPass(iData)
		cOut=self.calcLogProb(lOut[-1])
		cOut[cOut==0]=np.finfo(np.float64).tiny
		#err=-np.sum(np.sum(tOut*(np.log(cOut)),axis=0))/nSamples
		err = -np.mean(tOut*(np.log(cOut)))
		return err		

	def targetOut(self,labels):
		return (labels==np.unique(labels)).astype(int)

	def test(self,tstData,actualTstLabel):
		if actualTstLabel.ndim==1:
			actualTstLabel.reshape(-1,1)
		noTstData=len(actualTstLabel)
		tmpTstData = tstData*np.tile(self.splW,(np.shape(tstData)[0],1))
		#tstDataS = tstData
		fOut=self.forwardPassPost(tmpTstData)[-1]
		fOut=self.calcLogProb(fOut)
		predictedTstLabel=(np.argmax(fOut,axis=1)).reshape(-1,1)
		misClassifiedD=[actualTstLabel[i] for i in range(noTstData) if actualTstLabel[i] != predictedTstLabel[i]]
		return predictedTstLabel,len(misClassifiedD)

	def preTrain(self,trData,trLabels,valData=[],valLabels=[],optimizationFuncName='scg',cvThreshold=0.1,windowSize=10,nItr=10,
		weightPrecision=0,errorPrecision=0,verbose=False,freezeLayerFlag=True):
		
		def costFunc(W):
			if self.errorFunc=='MSE':
				return(self.costFuncMSE(W,trData,tOut))
			elif self.errorFunc=='CE':
				return(self.costFuncCE(W,trData,tOut))
			else:
				print('Wrong error function')

		def calcValErr(W):
			if self.errorFunc=='MSE':
				return(self.calcMSE(W,valData,vOut))
			elif self.errorFunc=='CE':
				return(self.calcCE(W,valData,vOut))
			else:
				print('Wrong error function')

		def calcTrErr(W):
			if self.errorFunc=='MSE':
				return(self.calcMSE(W,trData,tOut))
			elif self.errorFunc=='CE':
				return(self.calcCE(W,trData,tOut))
			else:
				print('Wrong error function')

		def gradient(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(trData)
			if self.freezeLayerFlag == True:
				dWs=self.backwardPassFreezeLayer(self.calcError(lOut[-1],tOut),lOut)
			else:
				dWs=self.backwardPass(self.calcError(lOut[-1],tOut),lOut)
			return self.flattenD(dWs)
			
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)

		def selectData(residual):
			squaredRes = residual**2
			err = np.sum(squaredRes,1).reshape(-1,1)
			l = np.where(err>=(np.mean(err)+np.std(err)))[1]
			if len(l) == 0:
				l = np.where(err>=(np.mean(err)))[1]
			return trData[l],tOut[l]

		tOut=self.targetOut(trLabels)
		vOut=self.targetOut(valLabels)

		#Start training for one hidden layer at a time
		iDim=self.inputDim
		oDim=self.outputDim
		self.freezeLayerFlag = freezeLayerFlag
		for l in range(len(self.hLayer)):
			self.hlNo=l+1
			netLayer=[iDim,self.hLayer[l],oDim]
			W=self.initWeight(netLayer)
			self.lActFunc=self.actFunc[l]
			print('Pre-training layer:',netLayer[0],'-->[',netLayer[1],']-->',netLayer[2],' with activation function:',self.lActFunc,' No of training data:',len(trData))
			if self.freezeLayerFlag == True:
				if optimizationFuncName == 'scgWithErrorCutoff':
					scgresult=scg.scgWithErrorCutoff(self.flattenD(W), costFunc, gradient, calcTrErr, calcValErr, cvThreshold, windowSize,
									xPrecision = weightPrecision,fPrecision = errorPrecision,
									nIterations = self.nItr[l], iterationVariable = self.iteration, ftracep=True, verbose=verbose)
				elif optimizationFuncName == 'scgWithEarlyStop':
					scgresult=scg.scgWithEarlyStop(self.flattenD(W),costFunc, gradient, calcTrErr, calcValErr, calcValErr, windowSize,
									xPrecision=weightPrecision,fPrecision=errorPrecision,nIterations=self.nItr,
									iterationVariable=self.iteration,ftracep=True,verbose=verbose)
				else:					
					scgresult=scg.scg(self.flattenD(W), costFunc, gradient, calcTrErr, calcValErr, xPrecision=weightPrecision,
						fPrecision=errorPrecision,nIterations=self.nItr[l],iterationVariable=self.iteration,ftracep=True,verbose=verbose)
			else:
				scgresult=scg.scgWithErrorCutoff(self.flattenD(self.netW), costFunc, gradient, costFunc, calcValErr, cvThreshold, windowSize,
								xPrecision = weightPrecision,fPrecision = errorPrecision,nIterations = self.nItr[l],
								iterationVariable = self.iteration,ftracep=True,verbose=verbose)
			self.unFlattenD(scgresult['x'])
			self.layer_error_trace.append(scgresult['ftrace'])
			#self.layer_weight_trace.append(scgresult['weightNorm'])
			self.layer_iteration.append(scgresult['nIterations'])
			print('No of SCG iterations:',self.layer_iteration[-1])			
			iDim=self.hLayer[l]
		self.preTrainingDone=True		
		return self

	def postTrain(self,trData,trLabels,valData=[],valLabels=[],tstData=[],tstLabels=[],optimizationFuncName='scg',windowSize=0,nItr=10,batchFlag=False,batchSize=100,noEpoch=500,errThreshold=0):

		def costFunc(W):
			if self.errorFunc=='MSE':
				return(self.costFuncMSE(W,trData,tOut))
			elif self.errorFunc=='CE':
				return(self.costFuncCE(W,trData,tOut))
			else:
				print('Wrong error function')

		def costFuncSPL(W):
			#pdb.set_trace()
			self.splW = W[:self.inputDim]
			tmpTrData = trData*np.tile(self.splW,(np.shape(trData)[0],1))		
			if self.errorFunc=='MSE':
				return(self.costFuncMSE(W[self.inputDim:],tmpTrData,tOut))
			elif self.errorFunc=='CE':
				return(self.costFuncCE(W[self.inputDim:],tmpTrData,tOut))
			else:
				print('Wrong error function')

		def gradient(W):
			#pdb.set_trace()
			self.unFlattenD(W)
			lOut=self.forwardPassPost(trData)
			dWs=self.backwardPassPost(self.calcError(lOut[-1],tOut),lOut)
			return self.flattenD(dWs)

		def gradientSPL(W):
			#pdb.set_trace()
			self.splW = W[:self.inputDim]			
			self.unFlattenD(W[self.inputDim:])			
			tmpTrData = trData*np.tile(self.splW,(np.shape(trData)[0],1))
			lOut=self.forwardPassPost(tmpTrData)
			dWs=self.backwardPassSPL(self.calcError(lOut[-1],tOut),lOut,trData)
			#pdb.set_trace()
			#spldWs = dWs[0]
			if self.l1Penalty != None:
				signW = copy(self.splW)
				signW[np.where(signW<0)] = -1
				signW[np.where(signW>0)] = 1
			if self.l1Penalty != None and self.l2Penalty != None:
				dWs[0] = dWs[0] + self.l1Penalty*signW + self.l2Penalty*W[:self.inputDim]
			elif self.l1Penalty != None and self.l2Penalty == None:				
				dWs[0] = dWs[0] + self.l1Penalty*signW
			elif self.l1Penalty == None and self.l2Penalty != None:
				dWs[0] = dWs[0] + self.l2Penalty*W[:self.inputDim]
			elif self.WeightElimination != None:
				#pdb.set_trace()
				penaltyTerm = (2*self.WeightElimination[0]*self.WeightElimination[1]**2)*np.array([(self.splW[l]/((self.splW[l]**2+self.WeightElimination[1]**2)**2)) for l in range(len(self.splW))])
				dWs[0] = dWs[0] + penaltyTerm/(np.shape(trData)[0]*self.outputDim)
			else:
				return self.flattenD(dWs)
			#dWs[0] = tmpSPLdWs
			#print('Norm of derivatives of splW',np.linalg.norm(dWs[0]))
			return self.flattenD(dWs)

		def calcValErr(W):
			if self.errorFunc=='MSE':
				return(self.calcMSE(W,valData,vOut))
			elif self.errorFunc=='CE':				
				return(self.calcCE(W,valData,vOut))
			else:
				print('Wrong error function')

		def calcTrErr(W):
			if self.errorFunc=='MSE':
				return(self.calcMSE(W,trData,tOut))
			elif self.errorFunc=='CE':
				return(self.calcCE(W,trData,tOut))
			else:
				print('Wrong error function')

		def calcValErrSPL(W):
			self.splW = W[:self.inputDim]
			tmpValData = trData*np.tile(self.splW,(np.shape(valData)[0],1))
			if self.errorFunc=='MSE':
				return(self.calcMSE(W[self.inputDim:],tmpValData,vOut))
			elif self.errorFunc=='CE':				
				return(self.calcCE(W[self.inputDim:],tmpValData,vOut))
			else:
				print('Wrong error function')

		def calcTrErrSPL(W):
			self.splW = W[:self.inputDim]
			tmpTrData = trData*np.tile(self.splW,(np.shape(trData)[0],1))
			if self.errorFunc=='MSE':
				return(self.calcMSE(W[self.inputDim:],tmpTrData,tOut))
			elif self.errorFunc=='CE':
				return(self.calcCE(W[self.inputDim:],tmpTrData,tOut))
			else:
				print('Wrong error function')
				
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)		

		tOut=self.targetOut(trLabels)
		vOut=self.targetOut(valLabels)
		#vOut=valLabels		
		if batchFlag:
			trDataOrg=copy(trData)
			tOutOrg = copy(tOut)
		self.freezeLayerFlag = False
		self.runningPostTraining = True
		self.iteration = nItr		
		self.lActFunc=self.actFunc

		if self.preTrainingDone==False:
			netLayer=[]
			netLayer.append(self.inputDim)
			netLayer.extend(self.hLayer)
			netLayer.extend([self.outputDim])
			self.initWeight(netLayer)
		self.splW = np.ones([1,self.inputDim])	
		#print('Post training network',self.inputDim,'-->',self.hLayer,'-->',self.outputDim,'with SPL')

		if batchFlag == True:
			noBatches = int(len(trData)/batchSize)
			epochError = []
			for epoch in range(noEpoch):
				shuffledIndex = np.arange(len(trDataOrg))
				np.random.shuffle(shuffledIndex)
				#pdb.set_trace()
				batchIndex = shuffledIndex.reshape(noBatches,batchSize)				
				print('Epoch:',epoch+1)
				for batch in range(noBatches):
					#pdb.set_trace()
					batchError = []
					trData = trDataOrg[batchIndex[batch,:],:]
					tOut = tOutOrg[batchIndex[batch,:],:]
					if optimizationFuncName == 'scg':
						result=scg.scg(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr,xPrecision=[],fPrecision=[],
						nIterations=nItr,iterationVariable=nItr,ftracep=True,verbose=False)
					elif optimizationFuncName == 'cg':
						result=CGS(self.flattenD(self.netW),funcCG,[nItr],calcTrErr, calcValErr)
					self.unFlattenD(result['x'])
					batchError.append(result['ftrace'][-1])
					#pdb.set_trace()
				epochError.append(np.vstack((batchError)))				
				predictedTstLabel,noMistake = self.test(tstData,tstLabels)
				print('After epoch:',epoch+1,' No of misclassifications=',noMistake)
			self.postTrErr = np.vstack((epochError))
			print('Average epoch error:',np.mean(self.postTrErr))
		else:
			if optimizationFuncName == 'scgWithEarlyStop':
				result=scg.scgWithEarlyStop(self.flattenD(self.netW),costFunc,gradient,calcTrErr,calcValErr,windowSize,xPrecision=[],fPrecision=[],nIterations=nItr,
								iterationVariable=nItr,ftracep=True,verbose=False)
			elif optimizationFuncName == 'scgWithErrorCutoff':
				result=scg.scgWithErrorCutoff(self.flattenD(self.netW),costFunc,gradient,calcTrErr,calcValErr,errThreshold,windowSize,xPrecision=[],fPrecision=[],nIterations=nItr,
								iterationVariable=nItr,ftracep=True,verbose=False)
			elif optimizationFuncName == 'cg':
				result=CGS(self.flattenD(self.netW),funcCG,[nItr],calcTrErr, calcValErr)
			else:
				#pdb.set_trace()
				print('Post training network',self.inputDim,'-->',self.hLayer,'-->',self.outputDim,'without L1/L2 penalty')
				tmpPenalty=self.l1Penalty
				self.l1Penalty=None
				result1=scg.scg(np.append(self.splW,self.flattenD(self.netW)), costFuncSPL, gradientSPL, calcTrErrSPL,calcValErrSPL, xPrecision=[],fPrecision=[],nIterations=50,iterationVariable=nItr,
					ftracep=True,verbose=False)
				self.splW = result1['x'][:self.inputDim]
				self.unFlattenD(result1['x'][self.inputDim:])
				self.splWBeforePenalty = copy(self.splW)
				#pdb.set_trace()
				print('Post training network',self.inputDim,'-->',self.hLayer,'-->',self.outputDim,'with L1/L2 penalty')
				self.l1Penalty=tmpPenalty
				result=scg.scg(np.append(self.splW,self.flattenD(self.netW)), costFuncSPL, gradientSPL, calcTrErrSPL,calcValErrSPL, xPrecision=[],fPrecision=[],nIterations=nItr,iterationVariable=nItr,
					ftracep=True,verbose=False)
			self.splW = result['x'][:self.inputDim]
			self.unFlattenD(result['x'][self.inputDim:])
			self.postTrErr = result['ftrace']
			#self.trErrorTrace = result['trMSE']
			self.valErrorTrace = result['valMSE']
			#self.iteration = result['bestItr']
			self.iterationWeights=result['weights']	
		print('Post training is done. No of iteration took:',self.iteration)
		#pdb.set_trace()
		predictedTrLabel,noErrTr = self.test(trData,trLabels)
		print('No of misclassification on training data',noErrTr)		
		predictedTstLabel,noErrTst = self.test(tstData,tstLabels)
		print('No of misclassification on test data',noErrTst)
		return self

	def postTrainWithPretrainedWs(self,trData,trLabels,valData=[],valLabels=[],tstData=[],tstLabels=[],optimizationFuncName='scg',windowSize=0,nItr=10,batchFlag=False,batchSize=100,noEpoch=500,errThreshold=0):

		def costFunc(W):
			if self.errorFunc=='MSE':
				return(self.costFuncMSE(W,trData,tOut))
			elif self.errorFunc=='CE':
				return(self.costFuncCE(W,trData,tOut))
			else:
				print('Wrong error function')

		def costFuncSPL(W):
			#pdb.set_trace()
			self.splW = W[:self.inputDim]
			tmpTrData = trData*np.tile(self.splW,(np.shape(trData)[0],1))		
			if self.errorFunc=='MSE':
				return(self.costFuncMSE(W[self.inputDim:],tmpTrData,tOut))
			elif self.errorFunc=='CE':
				return(self.costFuncCE(W[self.inputDim:],tmpTrData,tOut))
			else:
				print('Wrong error function')

		def gradient(W):
			#pdb.set_trace()
			self.unFlattenD(W)
			lOut=self.forwardPassPost(trData)
			dWs=self.backwardPassPost(self.calcError(lOut[-1],tOut),lOut)
			return self.flattenD(dWs)

		def gradientSPL(W):
			#pdb.set_trace()
			self.splW = W[:self.inputDim]			
			self.unFlattenD(W[self.inputDim:])			
			tmpTrData = trData*np.tile(self.splW,(np.shape(trData)[0],1))
			lOut=self.forwardPassPost(tmpTrData)
			dWs=self.backwardPassSPL(self.calcError(lOut[-1],tOut),lOut,trData)
			#pdb.set_trace()
			#spldWs = dWs[0]
			if self.l1Penalty != None:
				signW = copy(self.splW)
				signW[np.where(signW<0)] = -1
				signW[np.where(signW>0)] = 1
			if self.l1Penalty != None and self.l2Penalty != None:
				dWs[0] = dWs[0] + self.l1Penalty*signW + self.l2Penalty*W[:self.inputDim]
			elif self.l1Penalty != None and self.l2Penalty == None:				
				dWs[0] = dWs[0] + self.l1Penalty*signW
			elif self.l1Penalty == None and self.l2Penalty != None:
				dWs[0] = dWs[0] + self.l2Penalty*W[:self.inputDim]
			elif self.WeightElimination != None:
				#pdb.set_trace()
				penaltyTerm = (2*self.WeightElimination[0]*self.WeightElimination[1]**2)*np.array([(self.splW[l]/((self.splW[l]**2+self.WeightElimination[1]**2)**2)) for l in range(len(self.splW))])
				dWs[0] = dWs[0] + penaltyTerm/(np.shape(trData)[0]*self.outputDim)
			else:
				return self.flattenD(dWs)
			#dWs[0] = tmpSPLdWs
			#print('Norm of derivatives of splW',np.linalg.norm(dWs[0]))
			return self.flattenD(dWs)

		def calcValErr(W):
			if self.errorFunc=='MSE':
				return(self.calcMSE(W,valData,vOut))
			elif self.errorFunc=='CE':				
				return(self.calcCE(W,valData,vOut))
			else:
				print('Wrong error function')

		def calcTrErr(W):
			if self.errorFunc=='MSE':
				return(self.calcMSE(W,trData,tOut))
			elif self.errorFunc=='CE':
				return(self.calcCE(W,trData,tOut))
			else:
				print('Wrong error function')

		def calcValErrSPL(W):
			self.splW = W[:self.inputDim]
			tmpValData = trData*np.tile(self.splW,(np.shape(valData)[0],1))
			if self.errorFunc=='MSE':
				return(self.calcMSE(W[self.inputDim:],tmpValData,vOut))
			elif self.errorFunc=='CE':				
				return(self.calcCE(W[self.inputDim:],tmpValData,vOut))
			else:
				print('Wrong error function')

		def calcTrErrSPL(W):
			self.splW = W[:self.inputDim]
			tmpTrData = trData*np.tile(self.splW,(np.shape(trData)[0],1))
			if self.errorFunc=='MSE':
				return(self.calcMSE(W[self.inputDim:],tmpTrData,tOut))
			elif self.errorFunc=='CE':
				return(self.calcCE(W[self.inputDim:],tmpTrData,tOut))
			else:
				print('Wrong error function')
				
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)		

		tOut=self.targetOut(trLabels)
		vOut=self.targetOut(valLabels)
		#vOut=valLabels		
		if batchFlag:
			trDataOrg=copy(trData)
			tOutOrg = copy(tOut)
		self.freezeLayerFlag = False
		self.runningPostTraining = True
		self.iteration = nItr		
		self.lActFunc=self.actFunc

		#print('Post training network',self.inputDim,'-->',self.hLayer,'-->',self.outputDim,'with SPL')

		if batchFlag == True:
			noBatches = int(len(trData)/batchSize)
			epochError = []
			for epoch in range(noEpoch):
				shuffledIndex = np.arange(len(trDataOrg))
				np.random.shuffle(shuffledIndex)
				#pdb.set_trace()
				batchIndex = shuffledIndex.reshape(noBatches,batchSize)				
				print('Epoch:',epoch+1)
				for batch in range(noBatches):
					#pdb.set_trace()
					batchError = []
					trData = trDataOrg[batchIndex[batch,:],:]
					tOut = tOutOrg[batchIndex[batch,:],:]
					if optimizationFuncName == 'scg':
						result=scg.scg(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr,xPrecision=[],fPrecision=[],
						nIterations=nItr,iterationVariable=nItr,ftracep=True,verbose=False)
					elif optimizationFuncName == 'cg':
						result=CGS(self.flattenD(self.netW),funcCG,[nItr],calcTrErr, calcValErr)
					self.unFlattenD(result['x'])
					batchError.append(result['ftrace'][-1])
					#pdb.set_trace()
				epochError.append(np.vstack((batchError)))				
				predictedTstLabel,noMistake = self.test(tstData,tstLabels)
				print('After epoch:',epoch+1,' No of misclassifications=',noMistake)
			self.postTrErr = np.vstack((epochError))
			print('Average epoch error:',np.mean(self.postTrErr))
		else:
			if optimizationFuncName == 'scgWithEarlyStop':
				result=scg.scgWithEarlyStop(self.flattenD(self.netW),costFunc,gradient,calcTrErr,calcValErr,windowSize,xPrecision=[],
					fPrecision=[],nIterations=nItr,iterationVariable=nItr,ftracep=True,verbose=False)
			elif optimizationFuncName == 'scgWithErrorCutoff':
				result=scg.scgWithErrorCutoff(self.flattenD(self.netW),costFunc,gradient,calcTrErr,calcValErr,errThreshold,windowSize,xPrecision=[],
					fPrecision=[],nIterations=nItr,iterationVariable=nItr,ftracep=True,verbose=False)
			elif optimizationFuncName == 'cg':
				result=CGS(self.flattenD(self.netW),funcCG,[nItr],calcTrErr, calcValErr)
			else:
				#print('Finetuning network',self.inputDim,'-->',self.hLayer,'-->',self.outputDim,'with L1/L2 penalty')
				#pdb.set_trace()
				result=scg.scg(np.append(self.splW,self.flattenD(self.netW)), costFuncSPL, gradientSPL, calcTrErrSPL,calcValErrSPL, xPrecision=[],
				fPrecision=[],nIterations=nItr,iterationVariable=nItr,ftracep=True,verbose=False)
			self.splW = result['x'][:self.inputDim]
			self.unFlattenD(result['x'][self.inputDim:])
			self.postTrErr = result['ftrace']
			#self.trErrorTrace = result['trMSE']
			self.valErrorTrace = result['valMSE']
			#self.iteration = result['bestItr']
			self.iterationWeights=result['weights']	
		#print('Post training is done. No of iteration took:',self.iteration)
		#pdb.set_trace()
		#predictedTrLabel,noErrTr = self.test(trData,trLabels)
		#print('No of misclassification on training data',noErrTr)		
		#predictedTstLabel,noErrTst = self.test(tstData,tstLabels)
		#print('No of misclassification on test data',noErrTst)
		return self
