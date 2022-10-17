import pdb
import numpy as np
from utilityDBN import confusionMatrix
from centroidencoderclassifier import CentroidencoderClassifier
from ModelValidationClass import ModelValidationClass
from utilityDBN import confusionMatrix,multiclassConfusionMatrix,PCA,standardizeData,unStandardizeData
import calcom
import pathway_info
import matplotlib.pyplot as plt

def vizData(prjData,nOriginalData,nSyntheticData):
	plt.scatter(prjData[:nOriginalData,0],prjData[:nOriginalData,1],c='r',label='Original Symp. Data')
	plt.scatter(prjData[nOriginalData:,0],prjData[nOriginalData:,1],c='g',label='Synthetic Symp. Data')
	plt.legend()
	plt.show()

def loadPwData(pwNames,sympFlag,timeIndex):
	ccd = calcom.io.CCDataSet('/data3/darpa/calcom/geo_data_processing/ccd_gse_11study.h5')
	pathway_info.append_pathways_to_ccd(ccd)
	if sympFlag==True:
		idxsSymp = ccd.find_attrs_by_values([['StudyID',['gse73072']],['symptomatic',True],['time_id',timeIndex],['disease',['h1n1']]])#Only hour 5
		timeIdsSymp = ccd.get_attr_values('time_id', idx_list=idxsSymp)
		print('Symptomatic time Ids:',np.unique(timeIdsSymp))
		idxsControl = ccd.find_attrs_by_values([['StudyID',['gse73072']],['symptomatic',True],['time_id',np.arange(-40,1)],['disease',['h1n1']]])#Only hour -21 and 0
	else:
		idxsAsymp = ccd.find_attrs_by_values([['StudyID',['gse73072']],['symptomatic',False],['time_id',timeIndex],['disease',['h1n1']]])#Only hour 5	
		timeIdsAsymp = ccd.get_attr_values('time_id', idx_list=idxsAsymp)
		print('Asymptomatic time Ids:',np.unique(timeIdsAsymp))
		idxsControl = ccd.find_attrs_by_values([['StudyID',['gse73072']],['symptomatic',False],['time_id',np.arange(-40,1)],['disease',['h1n1']]])#Only hour -21 and 0
	timeIdsControl = ccd.get_attr_values('time_id', idx_list=idxsControl)
	print('Control time Ids:',np.unique(timeIdsControl))
	#pdb.set_trace()
	allPwData={}
	for pw in pwNames:
		control = ccd.generate_data_matrix(idx_list=idxsControl, feature_set=pw)
		labeledControl = np.hstack((control,np.zeros([len(control),1])))
		if sympFlag==True:
			symp = ccd.generate_data_matrix(idx_list=idxsSymp, feature_set=pw)
			labeledSymp = np.hstack((symp,np.ones([len(symp),1])))
			allPwData[pw]=np.vstack((labeledControl,labeledSymp))
		else:
			asymp = ccd.generate_data_matrix(idx_list=idxsAsymp, feature_set=pw)
			labeledAsymp = np.hstack((asymp,np.ones([len(asymp),1])))			
			allPwData[pw]=np.vstack((labeledControl,labeledAsymp))
		print('Dim. of pathway',pw,' is:',np.shape(control)[1])
	#pdb.set_trace()
	return allPwData

def loadControlSympAsympData(pwNames,timeIndex):
	ccd = calcom.io.CCDataSet('/data3/darpa/calcom/geo_data_processing/ccd_gse_11study.h5')
	pathway_info.append_pathways_to_ccd(ccd)
	idxsSymp = ccd.find_attrs_by_values([['StudyID',['gse73072']],['symptomatic',True],['time_id',timeIndex],['disease',['h1n1','h3n2']]])
	idxsAsymp = ccd.find_attrs_by_values([['StudyID',['gse73072']],['symptomatic',False],['time_id',timeIndex],['disease',['h1n1','h3n2']]])
	idxsControl = ccd.find_attrs_by_values([['StudyID',['gse73072']],['time_id',[-21,0]],['disease',['h1n1','h3n2']]])#Take both symp. and asymp. control at hr.=-21
	
	allPwData={}
	for pw in pwNames:
		control = ccd.generate_data_matrix(idx_list=idxsControl, feature_set=pw)
		labeledControl = np.hstack((control,np.zeros([len(control),1])))
		symp = ccd.generate_data_matrix(idx_list=idxsSymp, feature_set=pw)
		labeledSymp = np.hstack((symp,np.ones([len(symp),1])))
		asymp = ccd.generate_data_matrix(idx_list=idxsAsymp, feature_set=pw)
		labeledAsymp = np.hstack((asymp,2*np.ones([len(asymp),1])))
		allPwData[pw]=np.vstack((labeledControl,labeledSymp,labeledAsymp))
	return allPwData

def extractData(data,label,extractLabel):	
	indices=np.where(label==extractLabel)[0]
	return data[indices,:],label[indices,:]

topPwNames=['PID_IL4_2PATHWAY', 'REACTOME_SPHINGOLIPID_METABOLISM', 'GSE29618_BCELL_VS_PDC_DAY7_FLU_VACCINE_UP','PID_AP1_PATHWAY', 'REACTOME_ACTIVATED_AMPK_STIMULATES_FATTY_ACID_OXIDATION_IN_MUSCLE',
 'GSE29617_CTRL_VS_DAY7_TIV_FLU_VACCINE_PBMC_2008_UP', 'REACTOME_ADAPTIVE_IMMUNE_SYSTEM', 'GSE29618_BCELL_VS_MONOCYTE_DAY7_FLU_VACCINE_DN', 'GSE29618_MONOCYTE_VS_PDC_DAY7_FLU_VACCINE_DN',
 'KEGG_ADIPOCYTOKINE_SIGNALING_PATHWAY', 'REACTOME_KERATAN_SULFATE_BIOSYNTHESIS', 'REACTOME_OXYGEN_DEPENDENT_PROLINE_HYDROXYLATION_OF_HYPOXIA_INDUCIBLE_FACTOR_ALPHA',
 'REACTOME_THE_ROLE_OF_NEF_IN_HIV1_REPLICATION_AND_DISEASE_PATHOGENESIS', 'BIOCARTA_EGFR_SMRTE_PATHWAY', 'PID_NEPHRIN_NEPH1_PATHWAY']
hLayerList=[[20,10,20],[20,10,20],[25,15,25],[20,10,20],[5],[25,10,25],[25,10,25],[20,10,20],[30,10,30],[20,10,20],[10,5,10],[5],[10,5,10],[10,5,10],[10,5,10]]
sympFlag=False
timeIndex=np.arange(1,6)#For hr. 5 data
#timeIndex=29#For hr. 29 data
#allPwData=loadPwData(topPwNames,sympFlag,timeIndex)
allPwData=loadControlSympAsympData(topPwNames,timeIndex)
pwName=[]
allBSR=[]
allSR=[]
index=0
useSyntheticData = True
syntheticSympHr5Data={}
for pw in topPwNames:
	pwName.append(pw)
	pwBSR,pwSR=[],[]
	for r in range(10):#repeat each exp 10 times and then take the average BSR		
		pwData = allPwData[pw]#labeled data; last col. is the label
		data,label=pwData[:,:-1],pwData[:,-1].reshape(-1,1)
		print('Running experiment for pathway:',pw,'. No of sample in pathway:',len(label))
		#classList = ['control5hr','asymp5hr']
		ce=CentroidencoderClassifier()
		hLayer = hLayerList[index]
		actFuncList = ['tanh' for hl in hLayer]
		errorFunc = 'MSE'
		optimizationFuncName='scg'
		noItrPre,noItrPost,noItrSoftmax=15,150,25
		noItrFinetune,batchFlag=50,'False'
		repeatCE = 1 #Param to enlarge training dataset by adding ce output with the original training data.
		
		ce.initParams(hLayer,actFuncList,errorFunc,optimizationFuncName,noItrPre,noItrPost,noItrSoftmax,noItrFinetune,batchFlag,repeatCE,useSyntheticData)
		#Run PCA to see how do the synthetic data look like in 2D PCA compare to original data
		'''
		D,L = extractData(data,label,extractLabel=1)#Pull off the symptomatic samples
		mu,std,standardizeD = standardizeData(D)
		syntheticData,syntheticLabel=ce.generateData(standardizeD,L,int(len(D)/2),1)#Only create sythetic data for symptomatic samples
		syntheticData = unStandardizeData(syntheticData,mu,std)

		D1,L1 = extractData(data,label,extractLabel=2)#Pull off the asymptomatic samples
		mu1,std1,standardizeD1 = standardizeData(D1)
		syntheticData1,syntheticLabel1=ce.generateData(standardizeD1,L1,int(len(D1)/2),1)#Only create sythetic data for asymptomatic samples
		syntheticData1 = unStandardizeData(syntheticData1,mu1,std1)
		syntheticData,syntheticLabel = np.vstack((syntheticData,syntheticData1)),np.vstack((syntheticLabel,syntheticLabel1))
		syntheticSympHr5Data[pw] = syntheticData
		'''
		#projectedData,eVals,eVecs=PCA(np.vstack((D,syntheticData)),2)
		#vizData(projectedData,len(D),len(syntheticData))
		#pdb.set_trace()
		
		expDetails = {}		
		'''
		#For random sub sampling 
		expDetails['expType'] = 'RandomSubSampling'
		expDetails['repetition'] = 10
		expDetails['split_ratio'] = 0.8
		mv = ModelValidationClass(data,label,ce,expDetails)
		predictedValLabel,modelParam = mv.runValidation()
		'''
		#For cross validation
		minorityClass = [1,2]		
		expDetails['expType'] = 'CrossValidation'
		expDetails['no_fold'] = 5
		expDetails['repetition'] = None
		expDetails['useSyntheticData'] = useSyntheticData
		if useSyntheticData==False:
			mv = ModelValidationClass(data,label,ce,expDetails)
		else:
			mv = ModelValidationClass(data,label,ce,expDetails,minorityClass)
		allPredictedLabels,allModelParams,results = mv.runValidation()
		pwBSR.append(np.mean(results['allBSR']))
		pwSR.append(np.mean(results['allSR']))
		#pdb.set_trace()
		
	allBSR.append(pwBSR)
	allSR.append(pwSR)
	index+=1
print('\n')
if useSyntheticData ==False:
	print('Experiment done without symthetic data.')
else:
	print('Experiment done with synthetic data.')
print('No. of controls:',len(np.where(label==0)[0]),' No. of symtomatic:',len(np.where(label==1)[0]),'No. of asymptomatic:',len(np.where(label==2)[0]),)
for p in range(len(topPwNames)):	
	print('Pathway:',topPwNames[p],' BSR:','{0:.2f}'.format(np.mean(allBSR[p])),'+/-','{0:.2f}'.format(np.std(allBSR[p])))
