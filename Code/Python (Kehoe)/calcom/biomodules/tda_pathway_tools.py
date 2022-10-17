import sys
#sys.path.append('/data3/darpa/calcom/modules/') #just in case
#sys.path.append('/home/katrina/a/aminian/multipripser/')

try:
    import calcom
except:
    sys.path.append('/data3/darpa/calcom')
    import calcom
#

import multiprocessing
import pathway_info as pi
import numpy as np

##########################################
#
# Parameters
#

data_prefix = '/data3/darpa/omic_databases/'
num_procs = 16

##########################################

q0 = {'time_id':np.arange(-100,1)}
q1 = {'time_id':np.arange(1,121),'shedding':True}
ccd = calcom.io.CCDataSet(data_prefix+'ccd_gse73072_geneid.h5')

idx1 = ccd.find(q1)
idx0 = ccd.find(q0)
idx = np.hstack( (idx0,idx1) )

stids = ccd.get_attr_values('StudyID',idx=idx)

subjids = ccd.get_attr_values('SubjectID',idx=idx)
data_pre = ccd.generate_data_matrix(idx=idx)

data = calcom.utils.limma(data_pre,subjids)

def identify_gene_ptrs(glist):
    import numpy as np
    gnames = np.array( ccd.variable_names )
    ptrs = []
    for g in glist:
        glocs = np.where(g==gnames)[0]
        if len(glocs)>0:
            ptrs.append(glocs[0])
    #
    return np.array(ptrs)
    #
#

def build_empty_pathway_dict():
    '''
    Does similar to process_one, but doesn't actually 
    do the TDA/barcode stuff. Useful for my other code 
    where I only care about the pointers for genes in the 
    pathways associated with the dataset.
    '''
    result = {}
    for mypw in pi.pathways:
        entry = {}
        entry['genes'] = mypw.genes

        ptrs = identify_gene_ptrs(mypw.genes)
        entry['pointers'] = ptrs
        
        result[mypw.name] = entry
    #
    return result
#

def process_one(mypw):
    import ripser_interface as ri
    entry = {}
    entry['genes'] = mypw.genes

    ptrs = identify_gene_ptrs(mypw.genes)
    entry['pointers'] = ptrs

    data_sub = data[:,ptrs]
    ph_result = ri.run_ripser_sim(data_sub, ripser_loc='/home/katrina/a/aminian/ripser/ripser')
    ph0 = ph_result[0][:-1]
    ph1 = ph_result[1]
    entry['raw_ph'] = ph_result
    entry['ph0_bars'] = ph0
    entry['ph1_bars'] = ph1
    entry['ph0_sum'] = np.sum( np.diff(ph0,axis=1) )
    entry['ph1_sum'] = np.sum( np.diff(ph1,axis=1) )
    print(mypw.name, entry['ph1_sum'])
    return entry
#

def pathway_barcodes():
    p = multiprocessing.Pool(num_procs)
    results = p.map(process_one, pi.pathways)
    output = {pi.pathways[i].name : results[i] for i in range(len(results))}
    return output
#

if __name__=="__main__":
    p = multiprocessing.Pool(num_procs)
    results = pathway_barcodes()
