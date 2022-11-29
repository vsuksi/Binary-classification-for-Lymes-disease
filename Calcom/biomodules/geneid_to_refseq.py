'''
Purpose: to modularize the initial loading 
of conversion script from geneid to refseq formats.
'''

import csv

prefix = '/data3/darpa/omics_databases/'

filename = 'geneid2refseq_human_rna.csv'

with open(prefix+filename,'r') as f:
    csvr = csv.reader(f, delimiter=',')
    _gid2rnaseq = list(csvr)
#

g2r = {e[0]:e[1:] for e in _gid2rnaseq}

# create the inverse dictionary.
# as it happens, this is single-valued.
# only geneid-to-refseq is multiply valued.
r2g = {}
for k,v in g2r.items():
    for e in v:
        if e in r2g:
            print('DANGER DANGER',e,':',k)
        r2g[e] = k
#

# May expand with more tools later.
