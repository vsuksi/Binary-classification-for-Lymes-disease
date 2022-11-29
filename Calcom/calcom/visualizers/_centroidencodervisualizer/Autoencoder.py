'''
oame:Tomojit Ghosh
Program Name: stackedAutoencoder.py
Purpose: For my MS thesis, I'm trying to training stacked autoencoder with freezing weight of outer layers.
''' 
import numpy as np
from copy import copy
from . import nonLinearOptimizationAlgorithms as opt
from scipy.special import expit
from . import utilityDBN as util
MSE = util.MSE


class BottleneckSAE:
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
		self.trained=False
		self.trErrorTrace=None
		self.valErrorTrace=None
		self.trMSETrace=None
		self.iteration=0
		self.layer_iteration = []
		self.hlNo=0
		self.layer_error_trace = []
		self.layer_weight_trace = []
		self.actFunc=netConfig['actFunc']#This will contain the activation functions of all the hidden layers of parallel layers
		self.errorFunc=netConfig['errorFunc']
		self.lActFunc=''
		self.nItr=netConfig['nItr']
		self.nPostItr=None
		self.nPosBatchtItr=None
		self.freezeLayerFlag = ''
		self.l1Penalty=None
		self.l2Penalty=None
		self.trSetSize=None
		self.trBatchSize=None
		self.preTrainingDone=False
		self.postTrainingDone=False

	def initWeight(self,nUnits):
		nLayers=len(nUnits)-1
		#W=[np.random.uniform(-np.sqrt(3),np.sqrt(3), size=(1+nUnits[i],nUnits[i+1])) / np.sqrt(nUnits[i]) for i in range(nLayers)]
		#W=[np.random.uniform(-1,1, size=(1+nUnits[i],nUnits[i+1])) / np.sqrt(nUnits[i]) for i in range(nLayers)]
		W=[0.1*np.random.normal(0,1, size=(1+nUnits[i],nUnits[i+1])) for i in range(nLayers)]
		if self.netW==[]:
			self.netW=[w for w in W]
		else:
			tmpW=[]
			tmpW=[np.squeeze(self.netW[i]) for i in range(int(len(self.netW)/2))]
			tmpW.extend([W[i] for i in range(len(W))])
			tmpW.extend([self.netW[i] for i in range(int(len(self.netW)/2),int(len(self.netW)),1)])
			self.netW=tmpW
		return W

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
		#if self.hlNo==2:
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
		#pdb.set_trace()
		lOut=[D]
		lLength=len(self.netW)
		for j in range(lLength):
			d=np.dot(lOut[-1],self.netW[j][1:,:])+self.netW[j][0]#first row in the weight is the bias
			#Take the activation function from the dictionary and apply it
			lOut.append(self.feval('self.'+self.lActFunc,d) if j<lLength-1 else d)
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
			elif 'rect' in self.lActFunc:
				derivatives = 1*np.array(lO[l]>=0).astype(int)
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'rectl' in self.lActFunc:
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				#pdb.set_trace()
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
		#pdb.set_trace()
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
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
				#delta=lO[l]*(np.dot(deltas[-1],self.netW[l][1:,:].T))
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

	def regenD(self,X):
		Xst=self.standardizeX(X)
		Zs = self.forwardPass(Xst)
		Zs[-1][:] = self.unstandardizeT(Zs[-1][:])
		return Zs
		
	def regenDWOStandardize(self,X):		
		Zs = self.forwardPass(X)
		Zs[-1][:] = Zs[-1][:]
		return Zs
		
	def calcLogProb(self,cOut):
		'''		
		target=np.exp(cOut)
		denom=np.repeat(np.sum(target,1).reshape(-1,1),self.outputDim,axis=1)
		return target/denom
		'''		
		#Trick to avoid Overflow using Chuck's code
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
		
	def train(self,iData,oData,valInput,valOutput,optimizationFuncName,cvThreshold,windowSize,nItr=10,weightPrecision=0,errorPrecision=0,verbose=False,freezeLayerFlag=True):
		
		def calcError(cOut):
			err=(cOut-oData)/(np.shape(oData)[0]*self.outputDim)			
			return err

		def costFunc(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			if self.errorFunc=='MSE':
				return 0.5 * np.mean((lOut[-1] - oData)**2)
			elif self.errorFunc=='CE':
				cOut=self.calcLogProb(lOut[-1])
				cOut[cOut==0]=np.finfo(np.float64).tiny
				return -np.mean(oData*np.log(cOut)+(1-oData)*np.log(1-cOut))
			else:
				print('Wrong cost function')
				
		def gradient(W):			
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			if self.freezeLayerFlag == True:
				dWs=self.backwardPassFreezeLayer(calcError(lOut[-1]),lOut)
			else:
				dWs=self.backwardPass(calcError(lOut[-1]),lOut)
			return self.flattenD(dWs)
			
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)
			
		def calcTrErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,iData,oData)
			elif self.errorFunc == 'CE':
				return calcCE(W,iData,oData)
			
		def calcValErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,valInput,valOutput)
			elif self.errorFunc == 'CE':
				return calcCE(W,valInput,valOutput)
		
		def calcCE(W,inputData,outputData):
			#This function will return calculated per sample per output dimension
			self.unFlattenD(W)
			nSamples=np.shape(inputData)[0]
			lOut = self.forwardPass(inputData)
			cOut=lOut[-1]
			cOut[cOut==0]=np.finfo(np.float64).tiny		
			err = -np.mean(outputData*np.log(cOut)+(1-outputData)*np.log(1-cOut))
			return err

		def calcMSE(W,inputData,outputData):
			##This function will return the RMSE on training data. The error is calculated per data per output dimension
			self.unFlattenD(W)
			lOut = self.forwardPass(inputData)
			squaredRes = (lOut[-1] - outputData)**2
			rmse = np.sqrt(np.mean(squaredRes))
			return rmse
		
		#Start training for one hidden layer at a time
		iDim=self.inputDim
		oDim=self.outputDim
		self.freezeLayerFlag = freezeLayerFlag
		for l in range((np.ceil(len(self.hLayer)/2)).astype(int)):
			self.hlNo=l+1
			netLayer=[iDim,self.hLayer[l],oDim]
			W=self.initWeight(netLayer)
			self.lActFunc=self.actFunc[l]
			print('Training layer:',str(netLayer),' with activation function:',self.lActFunc)
			if self.freezeLayerFlag == True:
				if optimizationFuncName == 'scgWithErrorCutoff':
					scgresult=opt.scgWithErrorCutoff(self.flattenD(W), costFunc, gradient, calcTrErr,calcValErr,cvThreshold, windowSize,
									xPrecision = weightPrecision,fPrecision = errorPrecision,
									nIterations = self.nItr[l], iterationVariable = self.iteration, ftracep=True, verbose=verbose)
				else:
					scgresult=opt.scg(self.flattenD(W), costFunc, gradient, calcTrErr, calcValErr, xPrecision=weightPrecision,
							fPrecision=errorPrecision,nIterations=self.nItr[l],iterationVariable=self.iteration,ftracep=True,verbose=verbose)
			else:
				if optimizationFuncName == 'scgWithErrorCutoff':
					scgresult=opt.scgWithErrorCutoff(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, cvThreshold, windowSize,
									xPrecision = weightPrecision,fPrecision = errorPrecision,
									nIterations = self.nItr[l],iterationVariable = self.iteration, ftracep=True, verbose=verbose)
				else:
					scgresult=opt.scg(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr,xPrecision=weightPrecision,
							fPrecision=errorPrecision,nIterations=self.nItr[l],iterationVariable=self.iteration,ftracep=True,verbose=verbose)
			self.unFlattenD(scgresult['x'])
			self.layer_error_trace.append(scgresult['ftrace'])
			#self.layer_error_trace.append(scgresult['trMSE'])
			#self.layer_weight_trace.append(scgresult['weightNorm'])
			self.layer_iteration.append(scgresult['nIterations'])
			print('No of SCG iterations:',self.layer_iteration[-1])
			iDim=oDim=self.hLayer[l]
			#for w in self.netW:
			#	print(np.linalg.norm(w))
		return self

class BottleneckAE(BottleneckSAE):
	def __init__(self,netConfig):
		BottleneckSAE.__init__(self,netConfig)
		netLayer = copy(netConfig['hL'])
		netLayer.insert(0,netConfig['inputL'])
		netLayer.insert(len(netLayer),netConfig['outputL'])
		self.netW = BottleneckSAE.initWeight(self,netLayer)
		self.validationError = []
		self.trW = None

	def unFlattenD(self,Ws):
		#print('No of zeros',len(np.where(Ws==0)[0]))
		sIndex=0
		tmpWs=[]
		Ws=Ws.reshape(1,-1)
		for i in range(len(self.netW)):
			d1=np.shape(self.netW[i])[0]
			d2=np.shape(self.netW[i])[1]
			self.netW[i]=(Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2)
			sIndex=sIndex+(d1*d2)

	def returnUnFlattenD(self,Ws):
		sIndex=0
		tmpWs=[]
		Ws=Ws.reshape(1,-1)
		tmpW=[]
		for i in range(len(self.netW)):
			d1=np.shape(self.netW[i])[0]
			d2=np.shape(self.netW[i])[1]
			tmpW.append((Ws[0,sIndex:sIndex+d1*d2]).reshape(d1,d2))
			sIndex=sIndex+(d1*d2)
		return tmpW

	def forwardPass(self,D):
		#if self.__class__.__name__=='AE':
			#pdb.set_trace()
		#This function will return the network output for each layer.'key' is the identifier for each layer
		#print ('Class Name: ',self.__class__.__name__)
		#print('Calling forwardPass from',self.__class__.__name__)
		lOut=[D]
		lLength=len(self.netW)
		#pdb.set_trace()
		for j in range(lLength):
			d=np.dot(lOut[-1],self.netW[j][1:,:])+self.netW[j][0]#first row in the weight is the bias
			#Take the activation function from the dictionary and apply it
			#print('Act func.',self.actFunc[j])
			lOut.append(self.feval('self.'+self.actFunc[j],d))
		return lOut
	
	def backwardPass(self,error,lO):
		#This will return the partial derivatives for all the layers.
		deltas=[error]
		for l in range(len(self.netW)-1,0,-1):
			self.lActFunc=self.actFunc[l-1]
			#print('Act func.',self.lActFunc)
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
				#pdb.set_trace()
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

	def minErrorClassification(self,trData,trLabels,tstData,tstLabels):
		if trLabels.ndim==1:
			trLabels=trLabels.reshape(-1,1)
		if tstLabels.ndim==1:
			tstLabels=tstLabels.reshape(-1,1)
		noTstData=np.shape(tstLabels)[0]
		noTrData = np.shape(trLabels)[0]
		predictedTr=self.forwardPass(self.standardizeX(trData))[-1]#reconstructed input data
		predictedTst=self.forwardPass(self.standardizeX(tstData))[-1]
		predictedLabels=[]
		for d in range(noTstData):
			tmpData = np.tile(predictedTst[d,:],(noTrData,1))
			squaredRes = np.sum((predictedTr - tmpData)**2,axis=1)
			predictedLabels.append(trLabels[np.where(squaredRes==np.min(squaredRes))[0][0]])
		predictedLabels=np.vstack((predictedLabels))
		misClassifiedD=[tstLabels[i] for i in range(noTstData) if tstLabels[i] != predictedLabels[i]]
		return predictedLabels,len(misClassifiedD)

	def avgErrorClassification(self,trData,trLabels,tstData,tstLabels):
		if trLabels.ndim==1:
			trLabels=trLabels.reshape(-1,1)
		if tstLabels.ndim==1:
			tstLabels=tstLabels.reshape(-1,1)
		noTstData=np.shape(tstLabels)[0]
		noTrData = np.shape(trLabels)[0]
		cLabels=np.unique(trLabels)
		predictedTr=self.forwardPass(self.standardizeX(trData))[-1]#reconstructed input data
		predictedTst=self.forwardPass(self.standardizeX(tstData))[-1]
		predictedLabels=[]
		for d in range(noTstData):
			mse=[]
			for c in cLabels:
				indexSet=np.where(trLabels==c)[0]
				tmpData = np.tile(predictedTst[d,:],(len(indexSet),1))
				mse.append(np.mean((predictedTr[indexSet,:] - tmpData)**2))
			mse=np.vstack((mse))
			predictedLabels.append(np.where(mse==np.min(mse))[0][0])
		misClassifiedD=[tstLabels[i] for i in range(noTstData) if tstLabels[i] != predictedLabels[i]]
		return predictedLabels,len(misClassifiedD)

	def representativeClassification(self,classValues,tstData,tstLabels):
		if tstLabels.ndim==1:
			tstLabels=tstLabels.reshape(-1,1)
		noTstData=np.shape(tstLabels)[0]
		noClasses=np.shape(classValues)[0]
		classLabels=np.arange(noClasses)
		predictedTst=self.forwardPass(self.standardizeX(tstData))[-1]
		predictedLabels=[]
		for d in range(noTstData):
			tmpData = np.tile(predictedTst[d,:],(noClasses,1))
			squaredRes = np.sum((classValues - tmpData)**2,axis=1)
			predictedLabels.append(classLabels[np.where(squaredRes==np.min(squaredRes))[0][0]])
		predictedLabels=np.vstack((predictedLabels))
		misClassifiedD=[tstLabels[i] for i in range(noTstData) if tstLabels[i] != predictedLabels[i]]
		return predictedLabels,len(misClassifiedD)
	
	def train(self,iData,oData,valInput,valOutput,optimizationFuncName,windowSize=0,nItr=10,dropPercentage=0,weightPrecision=0,errorPrecision=0,verbose=False):#autoencoders own train function

		def calcError(cOut):
			err=(cOut-oData)/(np.shape(oData)[0]*self.outputDim)			
			return err

		def costFunc(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			if self.errorFunc=='MSE':
				if self.l1Penalty != None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/self.trSetSize)*np.sum(np.abs(W)) + (self.l2Penalty/(2*self.trSetSize))*np.sum(W**2)
				elif self.l1Penalty != None and self.l2Penalty == None:				
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/self.trSetSize)*np.sum(np.abs(W))
				elif self.l1Penalty == None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l2Penalty/(2*self.trSetSize))*np.sum(W**2)
				else:
					return 0.5 * np.mean((lOut[-1] - oData)**2)
			elif self.errorFunc=='CE':
				cOut=self.calcLogProb(lOut[-1])
				cOut[cOut==0]=np.finfo(np.float64).tiny
				return -np.mean(oData*np.log(cOut)+(1-oData)*np.log(1-cOut))
			else:
				print('Wrong cost function')

		def gradient(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			dWs=self.backwardPass(calcError(lOut[-1]),lOut)
			if self.l1Penalty != None:
				signW = copy(W)
				signW[np.where(signW<0)] = -1
				signW[np.where(signW>0)] = 1
			if self.l1Penalty != None and self.l2Penalty != None:
				return self.flattenD(dWs) + (self.l1Penalty/self.trSetSize)*signW + (self.l2Penalty/self.trSetSize)*W
			elif self.l1Penalty != None and self.l2Penalty == None:				
				return self.flattenD(dWs) + (self.l1Penalty/self.trSetSize)*signW
			elif self.l1Penalty == None and self.l2Penalty != None:
				return self.flattenD(dWs) + (self.l2Penalty/self.trSetSize)*W
			else:
				return self.flattenD(dWs)
				
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)
				
		def calcTrErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,iData,oData)
			elif self.errorFunc == 'CE':
				return calcCE(W,iData,oData)
			
		def calcValErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,valInput,valOutput)
			elif self.errorFunc == 'CE':
				return calcCE(W,valInput,valOutput)
				
		def calcCE(W,inputData,outputData):
			#This function will return calculated per sample per output dimension
			self.unFlattenD(W)
			nSamples=np.shape(inputData)[0]
			lOut = self.forwardPass(inputData)
			cOut=lOut[-1]
			cOut[cOut==0]=np.finfo(np.float64).tiny		
			err = -np.mean(outputData*np.log(cOut)+(1-outputData)*np.log(1-cOut))
			return err

		def calcMSE(W,inputData,outputData):
			#This function will return the RMSE on training data. The error is calculated per data per output dimension
			self.unFlattenD(W)
			lOut = self.forwardPass(inputData)
			squaredRes = (lOut[-1] - outputData)**2
			rmse = np.sqrt(np.mean(squaredRes))
			return rmse

		self.trSetSize=np.shape(iData)[0]
		'''
		print('Before post training...')
		for w in self.netW:
			print('Shape:', np.shape(w),'Norm ',np.linalg.norm(w))
		'''
		if optimizationFuncName == 'scgWithErrorCutoff':
			result=opt.scgWithErrorCutoff(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, cvThreshold, windowSize,
			xPrecision = weightPrecision,fPrecision = errorPrecision,nIterations = self.nItr[l], iterationVariable = self.iteration, ftracep=True, verbose=verbose)
		elif optimizationFuncName == 'scgWithEarlyStop':
			result=opt.scgWithEarlyStop(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, windowSize, xPrecision=weightPrecision,
			fPrecision=errorPrecision,nIterations=self.nItr,iterationVariable=self.iteration,ftracep=True,verbose=verbose)
		elif optimizationFuncName == 'scgWithDropConnect':
			result=opt.scgWithDropConnect(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, dropPercentage, xPrecision=weightPrecision,
			fPrecision=errorPrecision,nIterations=self.nItr,iterationVariable=self.iteration,ftracep=True,verbose=verbose)
		elif optimizationFuncName == 'scg':
			result=opt.scg(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, xPrecision=weightPrecision,
			fPrecision=errorPrecision,nIterations=self.nItr,iterationVariable=self.iteration,ftracep=True,verbose=verbose)
		else:
			result = CGS(self.flattenD(self.netW),funcCG,[self.nItr],calcTrErr, calcValErr)
		self.unFlattenD(result['x'])
		self.trErrorTrace = result['ftrace']
		self.valErrorTrace = result['valMSE']
		#self.trMSETrace = result['trMSE']
		self.iteration = result['nIterations']
		if optimizationFuncName == 'scg':
			self.trW = result['weights']
		print('No of SCG iteration:',self.iteration)
		'''
		print('Post training is done. No of iteration taken=',epoch)
		print('After post training...')
		for w in self.netW:
			print('Shape:', np.shape(w),'Norm ',np.linalg.norm(w))
		'''
		self.trained=True
		return self

	def batchTrain(self,iData,oData,valInput,valOutput,optimizationFuncName,windowSize=0,dropPercentage=0,nItr=10,batchSize=100,noEpoch=500):#autoencoders own train function

		def calcError(cOut):
			err=(cOut-oData)/(np.shape(oData)[0]*self.outputDim)			
			return err

		def costFunc(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			if self.errorFunc=='MSE':
				if self.l1Penalty != None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/(self.trSetSize*self.outputDim))*np.sum(np.abs(W)) + (self.l2Penalty/(2*self.trSetSize*self.outputDim))*np.sum(W**2)
				elif self.l1Penalty != None and self.l2Penalty == None:				
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/(self.trSetSize*self.outputDim))*np.sum(np.abs(W))
				elif self.l1Penalty == None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l2Penalty/(2*self.trSetSize*self.outputDim))*np.sum(W**2)
				else:
					return 0.5 * np.mean((lOut[-1] - oData)**2)
			elif self.errorFunc=='CE':
				cOut=self.calcLogProb(lOut[-1])
				cOut[cOut==0]=np.finfo(np.float64).tiny
				return -np.mean(oData*(np.log(cOut)))
			else:
				print('Wrong cost function')

		def gradient(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			dWs=self.backwardPass(calcError(lOut[-1]),lOut)
			if self.l1Penalty != None:
				signW = copy(W)
				signW[np.where(signW<0)] = -1
				signW[np.where(signW>0)] = 1
			if self.l1Penalty != None and self.l2Penalty != None:
				return self.flattenD(dWs) + (self.l1Penalty/(self.trSetSize*self.outputDim))*signW + (self.l2Penalty/(self.trSetSize*self.outputDim))*W
			elif self.l1Penalty != None and self.l2Penalty == None:				
				return self.flattenD(dWs) + (self.l1Penalty/(self.trSetSize*self.outputDim))*signW
			elif self.l1Penalty == None and self.l2Penalty != None:				
				return self.flattenD(dWs) + (self.l2Penalty/(self.trSetSize*self.outputDim))*W
			else:
				return self.flattenD(dWs)
				
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)

		def calcTrErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,iData,oData)
			elif self.errorFunc == 'CE':
				return calcCE(W,iData,oData)
			
		def calcValErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,valInput,valOutput)
			elif self.errorFunc == 'CE':
				return calcCE(W,valInput,valOutput)
				
		def calcCE(W,inputData,outputData):
			#This function will return calculated per sample per output dimension
			self.unFlattenD(W)
			nSamples=np.shape(inputData)[0]
			lOut = self.forwardPass(inputData)
			cOut=lOut[-1]
			cOut[cOut==0]=np.finfo(np.float64).tiny		
			err = -np.mean(outputData*np.log(cOut)+(1-outputData)*np.log(1-cOut))
			return err

		def calcMSE(W,inputData,outputData):
			#This function will return the RMSE on training data. The error is calculated per data per output dimension
			self.unFlattenD(W)
			lOut = self.forwardPass(inputData)
			squaredRes = (lOut[-1] - outputData)**2
			rmse = np.sqrt(np.mean(squaredRes))
			return rmse

		self.trSetSize=np.shape(iData)[0]
		iDataOrg=copy(iData)		

		noBatches = int(len(iData)/batchSize)
		epochTrError = []
		epochTrMSE = []
		epochValMSE = []
		orgTrSetSize = self.trSetSize
		self.trSetSize = batchSize
		for epoch in range(noEpoch):
			#pdb.set_trace()
			shuffledIndex = np.arange(orgTrSetSize)
			np.random.shuffle(shuffledIndex)
			batchIndex = shuffledIndex.reshape(noBatches,batchSize)
			print('Epoch:',epoch+1)
			for batch in range(noBatches):				
				batchTrError = []
				iData = iDataOrg[batchIndex[batch,:],:]
				oDataStandardize = oData1[batchIndex[batch,:],:]
				if optimizationFuncName == 'scgForBatch':
					result=opt.scgForBatch(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, xPrecision=0,
					fPrecision=0,nIterations=self.nItr,iterationVariable=self.iteration,ftracep=True,verbose=False)
				elif optimizationFuncName == 'cg':
					result=CGS(self.flattenD(self.netW),funcCG,[self.nPosBatchtItr],calcTrErr, calcValErr)					
				else:
					print('Wrong optimization function.')
				batchTrError.append(result['ftrace'][-1])
				#print('Batch:',batch+1,' is complete')
			epochTrError.append(np.mean(np.vstack((batchTrError))))
			iData = iDataOrg			
			epochTrMSE.append(calcTrMSE(self.flattenD(self.netW)))
			epochValMSE.append(calcValMSE(self.flattenD(self.netW)))
			print('After epoch:',epoch+1,'training MSE: ',epochTrMSE[-1])
			print('After epoch:',epoch+1,'validation MSE: ',epochValMSE[-1])
		self.unFlattenD(result['x'])
		self.trErrorTrace = np.vstack((epochTrError))
		#self.trMSETrace = np.vstack((epochTrMSE))
		self.valErrorTrace = np.vstack((epochValMSE))		
		print('Average error over epoch:',np.mean(self.trErrorTrace))
		#print('Average training MSE over epoch:',np.mean(self.trMSETrace))
		print('Average Validation MSE over epoch:',np.mean(self.valErrorTrace))
		self.trSetSize = orgTrSetSize
		self.trained=True
		return self
		
class Autoencoder(BottleneckSAE):
	def __init__(self,netConfig):
		BottleneckSAE.__init__(self,netConfig)
		
	def initWeight(self,nUnits):
		#pdb.set_trace()
		nLayers=len(nUnits)-1		
		#W=[np.random.uniform(-1,1, size=(1+nUnits[i],nUnits[i+1])) / np.sqrt(nUnits[i]) for i in range(nLayers)]
		W=[0.1*np.random.normal(0,1, size=(1+nUnits[i],nUnits[i+1])) for i in range(nLayers)]#Initializing weight from standard normal distribution
		if self.netW==[]:
			self.netW=[w for w in W]
		else:
			tmpW=[]
			self.netW[-1] = copy(W[0])
			self.netW.append(W[1])
		return W
		
	def forwardPass(self,D):
		#if self.__class__.__name__=='AE':
			#pdb.set_trace()
		#This function will return the network output for each layer.'key' is the identifier for each layer
		#print ('Class Name: ',self.__class__.__name__)
		#pdb.set_trace()
		lOut=[D]
		lLength=len(self.netW)
		for j in range(lLength):
			d=np.dot(lOut[-1],self.netW[j][1:,:])+self.netW[j][0]#first row in the weight is the bias
			#Take the activation function from the dictionary and apply it
			lOut.append(self.feval('self.'+self.actFunc[j],d) if j<lLength-1 else d)
		return lOut
		
	def backwardPassPost(self,error,lO):
		#This will return the partial derivatives for all the layers.
		#pdb.set_trace()
		deltas=[error]#added delta for the output layer. As output layer is linear so just the difference
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
			#Leaky rectifier linear unit function
			#Activation function: f(x)=dot(w.T,x) if dot(w.T,x) >0, otherwise 0.01*dot(w.T,x)
				derivatives = 0.01*np.array(lO[l]<0).astype(int)
				derivatives[derivatives==0] = 1
				delta=derivatives*(np.dot(deltas[-1],self.netW[l][1:,:].T))
				#delta=lO[l]*(np.dot(deltas[-1],self.netW[l][1:,:].T))
			elif 'linear' in lActFunc:
				delta=(np.dot(deltas[-1],self.netW[l][1:,:].T))
			else:
				print('Wrong activation function')
			deltas.append(delta)
		deltas.reverse()
		dWs=[]
		for l in range(len(self.netW)):
			dWs.append(np.vstack((deltas[l].sum(0),np.dot(lO[l].T, deltas[l]))))#The first row is the bias
		return dWs

	def minErrorClassification(self,trData,trLabels,tstData,tstLabels):
		if trLabels.ndim==1:
			trLabels=trLabels.reshape(-1,1)
		if tstLabels.ndim==1:
			tstLabels=tstLabels.reshape(-1,1)
		noTstData=np.shape(tstLabels)[0]
		noTrData = np.shape(trLabels)[0]		
		predictedTr=self.forwardPass(trData)[-1]#reconstructed input data		
		predictedTst=self.forwardPass(tstData)[-1]
		predictedLabels=[]
		for d in range(noTstData):
			tmpData = np.tile(predictedTst[d,:],(noTrData,1))
			squaredRes = np.sum((predictedTr - tmpData)**2,axis=1)
			predictedLabels.append(trLabels[np.where(squaredRes==np.min(squaredRes))[0][0]])
		predictedLabels=np.vstack((predictedLabels))
		misClassifiedD=[tstLabels[i] for i in range(noTstData) if tstLabels[i] != predictedLabels[i]]
		return predictedLabels,len(misClassifiedD)

	def avgErrorClassification(self,trData,trLabels,tstData,tstLabels):
		if trLabels.ndim==1:
			trLabels=trLabels.reshape(-1,1)
		if tstLabels.ndim==1:
			tstLabels=tstLabels.reshape(-1,1)
		noTstData=np.shape(tstLabels)[0]
		noTrData = np.shape(trLabels)[0]
		cLabels=np.unique(trLabels)		
		predictedTr=self.forwardPass(trData)[-1]#reconstructed input data		
		predictedTst=self.forwardPass(tstData)[-1]
		predictedLabels=[]
		for d in range(noTstData):
			mse=[]
			for c in cLabels:
				indexSet=np.where(trLabels==c)[0]
				tmpData = np.tile(predictedTst[d,:],(len(indexSet),1))
				mse.append(np.mean((predictedTr[indexSet,:] - tmpData)**2))
			mse=np.vstack((mse))
			predictedLabels.append(np.where(mse==np.min(mse))[0][0])
		misClassifiedD=[tstLabels[i] for i in range(noTstData) if tstLabels[i] != predictedLabels[i]]
		return predictedLabels,len(misClassifiedD)
	
	def preTrain(self,iData,oData,valInput,valOutput,optimizationFuncName='scg',cvThreshold=0.1,windowSize=10,nItr=10,weightPrecision=0,errorPrecision=0,verbose=False,freezeLayerFlag=True):
		
		def calcError(cOut):			
			err=(cOut-oData)/(np.shape(oData)[0]*self.outputDim)
			return err

		def costFunc(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			if self.errorFunc=='MSE':
				if self.l1Penalty != None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/self.trSetSize)*np.sum(np.abs(W)) + (self.l2Penalty/(2*self.trSetSize))*np.sum(W**2)
				elif self.l1Penalty != None and self.l2Penalty == None:				
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/self.trSetSize)*np.sum(np.abs(W))
				elif self.l1Penalty == None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l2Penalty/(2*self.trSetSize))*np.sum(W**2)
				else:
					return 0.5 * np.mean((lOut[-1] - oData)**2)
			elif self.errorFunc=='CE':
				cOut=self.calcLogProb(lOut[-1])
				cOut[cOut==0]=np.finfo(np.float64).tiny
				return -np.mean(oData*(np.log(cOut)))
			else:
				print('Wrong cost function')

		def gradient(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			if self.freezeLayerFlag == True:
				dWs=self.backwardPassFreezeLayer(calcError(lOut[-1]),lOut)
			else:
				dWs=self.backwardPass(calcError(lOut[-1]),lOut)
			if self.l1Penalty != None:
				signW = copy(W)
				signW[np.where(signW<0)] = -1
				signW[np.where(signW>0)] = 1
			if self.l1Penalty != None and self.l2Penalty != None:
				return self.flattenD(dWs) + (self.l1Penalty/(self.trSetSize*self.outputDim))*signW + (self.l2Penalty/(self.trSetSize*self.outputDim))*W
			elif self.l1Penalty != None and self.l2Penalty == None:				
				return self.flattenD(dWs) + (self.l1Penalty/(self.trSetSize*self.outputDim))*signW
			elif self.l1Penalty == None and self.l2Penalty != None:				
				return self.flattenD(dWs) + (self.l2Penalty/(self.trSetSize*self.outputDim))*W
			else:
				return self.flattenD(dWs)
				
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)
			
		def calcTrErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,iData,oData)
			elif self.errorFunc == 'CE':
				return calcCE(W,iData,oData)
			
		def calcValErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,valInput,valOutput)
			elif self.errorFunc == 'CE':
				return calcCE(W,valInput,valOutput)
		
		def calcCE(W,inputData,outputData):
			#This function will return calculated per sample per output dimension
			self.unFlattenD(W)
			nSamples=np.shape(inputData)[0]
			lOut = self.forwardPass(inputData)
			cOut=lOut[-1]
			cOut[cOut==0]=np.finfo(np.float64).tiny		
			err = -np.mean(outputData*np.log(cOut)+(1-outputData)*np.log(1-cOut))
			return err

		def calcMSE(W,inputData,outputData):
			##This function will return the RMSE on training data. The error is calculated per data per output dimension
			self.unFlattenD(W)
			lOut = self.forwardPass(inputData)
			squaredRes = (lOut[-1] - outputData)**2
			rmse = np.sqrt(np.mean(squaredRes))
			return rmse

		self.trSetSize=np.shape(iData)[0]
		#Start training for one hidden layer at a time
		iDim=self.inputDim
		oDim=self.outputDim
		self.freezeLayerFlag = freezeLayerFlag
		
		for l in range(len(self.hLayer)):
			self.hlNo=l+1
			netLayer=[iDim,self.hLayer[l],oDim]
			W=self.initWeight(netLayer)
			self.lActFunc=self.actFunc[l]
			print('Training layer:',str(netLayer),' with activation function:',self.lActFunc,' No of training data:',len(iData))
			if self.freezeLayerFlag == True:
				if optimizationFuncName == 'scgWithErrorCutoff':
					scgresult=opt.scgWithErrorCutoff(self.flattenD(W), costFunc, gradient, calcTrErr, calcValErr, cvThreshold, windowSize,
									xPrecision = weightPrecision,fPrecision = errorPrecision,
									nIterations = self.nItr[l], iterationVariable = self.iteration, ftracep=True, verbose=verbose)
				elif optimizationFuncName == 'scgWithEarlyStop':
					scgresult=opt.scgWithEarlyStop(self.flattenD(W),costFunc, gradient, calcTrErr, calcValErr, windowSize,
									xPrecision=weightPrecision,fPrecision=errorPrecision,nIterations=self.nItr,
									iterationVariable=self.iteration,ftracep=True,verbose=verbose)
				else:
					scgresult=opt.scg(self.flattenD(W), costFunc, gradient, calcTrErr, calcValErr, xPrecision=weightPrecision,
						fPrecision=errorPrecision,nIterations=self.nItr[l],iterationVariable=self.iteration,ftracep=True,verbose=verbose)
			else:
				scgresult=opt.scgWithErrorCutoff(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, cvThreshold, windowSize,
								xPrecision = weightPrecision,fPrecision = errorPrecision,nIterations = self.nItr[l],
								iterationVariable = self.iteration,ftracep=True,verbose=verbose)
			#pdb.set_trace()
			self.unFlattenD(scgresult['x'])
			self.layer_error_trace.append(scgresult['ftrace'])
			#self.layer_weight_trace.append(scgresult['weightNorm'])
			self.layer_iteration.append(scgresult['nIterations'])
			print('No of SCG iterations:',self.layer_iteration[-1])			
			iDim=self.hLayer[l]
		self.preTrainingDone=True
		return self
		
	def postTrain(self,iData,oData,valInput,valOutput,optimizationFuncName='scg',windowSize=0,nItr=10,batchFlag=False,batchSize=100,noEpoch=500,errThreshold=0):
		
		def calcError(cOut):			
			err=(cOut-oData)/(np.shape(oData)[0]*self.outputDim)
			return err
			
		def costFunc(W):
			#pdb.set_trace()
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			if self.errorFunc=='MSE':
				if self.l1Penalty != None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/self.trSetSize)*np.sum(np.abs(W)) + (self.l2Penalty/(2*self.trSetSize))*np.sum(W**2)
				elif self.l1Penalty != None and self.l2Penalty == None:				
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l1Penalty/self.trSetSize)*np.sum(np.abs(W))
				elif self.l1Penalty == None and self.l2Penalty != None:
					return 0.5 * np.mean((lOut[-1] - oData)**2) + (self.l2Penalty/(2*self.trSetSize))*np.sum(W**2)
				else:
					return 0.5 * np.mean((lOut[-1] - oData)**2)
			elif self.errorFunc=='CE':
				#cOut=self.calcLogProb(lOut[-1])
				pdb.set_trace()
				cOut=lOut[-1]
				cOut[cOut==0]=np.finfo(np.float64).tiny				
				return -np.mean(oData*np.log(cOut)+(1-oData)*np.log(1-cOut))
			else:
				print('Wrong cost function')				

		def gradient(W):
			self.unFlattenD(W)
			lOut=self.forwardPass(iData)
			#pdb.set_trace()
			dWs=self.backwardPassPost(calcError(lOut[-1]),lOut)
			if self.l1Penalty != None:
				signW = copy(W)
				signW[np.where(signW<0)] = -1
				signW[np.where(signW>0)] = 1
			if self.l1Penalty != None and self.l2Penalty != None:
				return self.flattenD(dWs) + (self.l1Penalty/(self.trSetSize*self.outputDim))*signW + (self.l2Penalty/(self.trSetSize*self.outputDim))*W
			elif self.l1Penalty != None and self.l2Penalty == None:				
				return self.flattenD(dWs) + (self.l1Penalty/(self.trSetSize*self.outputDim))*signW
			elif self.l1Penalty == None and self.l2Penalty != None:				
				return self.flattenD(dWs) + (self.l2Penalty/(self.trSetSize*self.outputDim))*W
			else:
				return self.flattenD(dWs)
				
		def funcCG(W):
			self.unFlattenD(W)
			err = costFunc(W)			
			dWs = gradient(W)
			return err,self.flattenD(dWs)
				
		def calcTrErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,iData,oData)
			elif self.errorFunc == 'CE':
				return calcCE(W,iData,oData)
			
		def calcValErr(W):
			if self.errorFunc == 'MSE':
				return calcMSE(W,valInput,valOutput)
			elif self.errorFunc == 'CE':
				return calcCE(W,valInput,valOutput)

		def calcCE(W,inputData,outputData):
			#This function will return calculated per sample per output dimension
			#pdb.set_trace()
			self.unFlattenD(W)
			nSamples=np.shape(inputData)[0]
			lOut = self.forwardPass(inputData)
			cOut=lOut[-1]
			cOut[cOut==0]=np.finfo(np.float64).tiny		
			err = -np.mean(outputData*np.log(cOut)+(1-outputData)*np.log(1-cOut))
			return err
				
		def calcMSE(W,inputData,outputData):
			#This function will return the RMSE on training data. The error is calculated per data per output dimension
			self.unFlattenD(W)
			cOut = self.forwardPass(inputData)
			noSamples = np.shape(cOut[-1])[0]			
			squaredRes = (cOut[-1] - outputData)**2
			rmse = np.sqrt(np.mean(squaredRes))
			return rmse

		if batchFlag:
			trInput=copy(iData)
			trOutput=copy(oData)
			self.nPosBatchtItr=nItr
		else:
			self.nPosItr = nItr
		self.trSetSize=np.shape(iData)[0]

		#If pretraining is done then don't need to initialize weights
		#pdb.set_trace()
		if self.preTrainingDone==False:
			netLayer=[]
			netLayer.append(self.inputDim)
			netLayer.extend(self.hLayer)
			netLayer.extend([self.outputDim])
			self.initWeight(netLayer)

		self.freezeLayerFlag = False
		if batchFlag == True:
			noBatches = int(len(iData)/batchSize)
			epochTrError = []
			epochValError = []
			for epoch in range(noEpoch):
				shuffledIndex = np.arange(len(trInput))
				np.random.shuffle(shuffledIndex)
				#pdb.set_trace()
				batchIndex = shuffledIndex.reshape(noBatches,batchSize)				
				print('Epoch:',epoch+1)
				batchTrError = []
				batchValError = []
				for batch in range(noBatches):
					#print('Batch:',batch+1)
					iData = trInput[batchIndex[batch,:],:]
					oData = trOutput[batchIndex[batch,:],:]					
					if optimizationFuncName == 'scgForBatch':
						result=opt.scgForBatch(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, xPrecision=0,
						fPrecision=0,nIterations=self.nItr,iterationVariable=self.iteration,ftracep=True,verbose=False)
					elif optimizationFuncName == 'cg':
						result=CGS(self.flattenD(self.netW),funcCG,[self.nPosBatchtItr],calcTrErr, calcValErr)
					else:
						result=opt.scg(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, xPrecision=[],fPrecision=[],nIterations=self.nPosBatchtItr,iterationVariable=self.nPosBatchtItr,
						ftracep=True,verbose=False)
					self.unFlattenD(result['x'])
					#batchTrError.append(result['ftrace'][-1])
					batchTrError.append(result['trMSE'])
					batchValError.append(result['valMSE'])
				batchTrError=np.vstack((batchTrError))
				batchValError=np.vstack((batchValError))
				avgBatchTrError=np.mean(batchTrError)
				avgBatchValError=np.mean(batchValError)
				print('After epoch ',epoch+1,' average training mse:',avgBatchTrError,' average validation mse:',avgBatchValError)
				epochTrError.append(avgBatchTrError)
				epochValError.append(avgBatchValError)
			self.postTrErr = np.vstack((epochTrError))
			print('Average epoch training error(ce/mse):',np.mean(self.postTrErr))
			print('Average epoch validation error(ce/mse):',np.mean(np.vstack((epochValError))))
		else:
			if optimizationFuncName == 'scgWithEarlyStop':
				result=opt.scgWithEarlyStop(self.flattenD(self.netW),costFunc,gradient,calcTrErr, calcValErr,windowSize,xPrecision=[],fPrecision=[],nIterations=nItr,
								iterationVariable=self.nPosItr,ftracep=True,verbose=False)
			elif optimizationFuncName == 'scgWithErrorCutoff':
				result=opt.scgWithErrorCutoff(self.flattenD(self.netW),costFunc,gradient,calcTrErr, calcValErr,errThreshold,windowSize,xPrecision=[],fPrecision=[],nIterations=nItr,
								iterationVariable=self.nPosItr,ftracep=True,verbose=False)
			else:
				result=opt.scg(self.flattenD(self.netW), costFunc, gradient, calcTrErr, calcValErr, xPrecision=[],fPrecision=[],nIterations=self.nPosItr,iterationVariable=nItr,
					ftracep=True,verbose=False)
			self.postTrErr = result['ftrace']
			#self.trErrorTrace = result['trMSE']
			self.valErrorTrace = result['valMSE']
			self.iteration = result['bestItr']						
			print('Post training is done. No of iteration took:',self.iteration)
		self.postTrainingDone=True
		return self
