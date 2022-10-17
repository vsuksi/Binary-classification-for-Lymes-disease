# Test datasets for merging

import calcom
import numpy as np

ccd0 = calcom.io.CCDataSet()
ccd1 = calcom.io.CCDataSet()

a_n = ['sex','time_id']

v_n0 = ['d','b','a']
v_n1 = ['b','q','d','n']

#
ccd0.add_attrs(a_n)
ccd1.add_attrs(a_n)

data0 = [[1,0,0],[0,1,1],[2,2,2]]
data1 = [[2,4,6,8],[-1,-2,-3,-4]]

# Make fake attributes
attrs0 = [ [np.random.choice(['m','f']), np.random.choice(24)] for _ in data0]
attrs1 = [ [np.random.choice(['m','f']), np.random.choice(24)] for _ in data1]

# Put in random data of the appropriate size
ccd0.add_datapoints(data0, a_n, attrs0)
ccd1.add_datapoints(data1, a_n, attrs1)

ccd0.add_variable_names(v_n0)
ccd1.add_variable_names(v_n1)

ccd0.add_feature_set('fset_0',[0,1])
ccd1.add_feature_set('fset_0',[2,3])
ccd1.add_feature_set('fset_1',[0,1,2])

# Make the ccg
ccg = calcom.io.CCGroup()

ccg.addCCDataSets([ccd0,ccd1],['ccd0','ccd1'])

# Tests!

print('\n\tTest 1: Do the maps in ccg.variableNameLocations\n\tcorrectly map to the originals?')
print('\t%s\n'%('-'*50))

new_v_n0 = [ccg.variable_names[i] for i in ccg.variableNameLocations['ccd0']]
new_v_n1 = [ccg.variable_names[i] for i in ccg.variableNameLocations['ccd1']]

print('\told: '+str(v_n0)+'; new:',str(list(new_v_n0)))
print('\told: '+str(v_n1)+'; new:',str(list(new_v_n1)))

v_eq_0 = np.array_equal(v_n0,new_v_n0)
v_eq_1 = np.array_equal(v_n1,new_v_n1)

print('\n\tccd0 variable names %s'%('agree' if v_eq_0 else 'disagree'))
print('\tccd1 variable names %s'%('agree' if v_eq_1 else 'disagree'))

print('\n\tTest 1 %s'%('passed' if (v_eq_0 and v_eq_1) else 'failed'))

print('\t'+('-'*50)+'\n\n')
print('\n\tTest 2: If we create data matrices restricting to the appropriate \n\tvariables and attributes, do we get the same data matrix?\n')
ccd2 = ccg.createCCDataSet()
idxs = ccd2.find_attr_by_value('source_dataset','ccd0')
v_n2 = ccd2.variable_names
i_n20 = []
for istr in v_n2:
    i = np.where(istr==np.array(ccd0.variable_names))[0]
    if len(i>0): i_n20.append(i[0])
#
data2 = ccd2.generate_data_matrix(idx_list=idxs)
data20 = ccd0.generate_data_matrix(feature_set=i_n20)

print('\tArrays: \n'+str(data2)+'; \n'+str(data20))
print('\n\t Test 2 %s'%('passed' if np.array_equal(data2,data20) else 'failed'))

print('\t'+('-'*50)+'\n\n')
print('\n\tTest 3: Do feature sets get appropriately mapped\n\twhen passed from the datasets to the group?\n')
print('\t____ not implemented on the group level yet ____')
