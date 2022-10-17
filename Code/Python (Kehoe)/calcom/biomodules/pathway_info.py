# figuring out how pathway information should be set up.

import calcom
import csv
import re
import numpy as np

# classes
class Pathway:
    '''
    A small object to contain a pathway (collection of genes)
    and corresponding metadata.
    The core of it should be a list of strings for genes along with a
    pathway name:
        self.name = kwargs.get('name','')
        self.genes = kwargs.get('genes',calcom.io.CCList())
    Other information we should get from the Broad Institute
    is the following:
        self.gene_format = kwargs.get('gene_format','gene_id')
        self.hyperlink = kwargs.get('hyperlink','')
        self.pw_type = kwargs.get('pw_type','') # canonical, biocarta, kegg, etc
    '''
    def __init__(self,**kwargs):
        self.gene_format = kwargs.get('gene_format','gene_id')
        self.hyperlink = kwargs.get('hyperlink','')
        self.pw_type = kwargs.get('pw_type','') # canonical, biocarta, kegg, etc
        self.genes = kwargs.get('genes',calcom.io.CCList())
        self.name = kwargs.get('name','')
    #
#

class Gene:
    '''
    A basic container for genes which keeps track of
    identifier, and which pathways it is present in.
    In the future, we'd also like to track different identifiers
    in different formats (Illumina, Entrez, etc)
    '''
    def __init__(self,**kwargs):
        self.format = kwargs.get('format','gene_id')
        # In future: get the plain name and its identifier
        # in multiple gene formats. Will make gene conversion
        # a little simpler and more coherent.
        self.pathways = kwargs.get('pathways',calcom.io.CCList())
    #
#

# Import pathway information
pw_prefix = '/data3/darpa/omics_databases/'
pw_files = ['c2.cp.biocarta.v6.1.entrez.gmt', 'c2.cp.kegg.v6.1.entrez.gmt', 'c2.cp.v6.1.entrez.gmt', 'c7.all.v6.1.entrez.gmt']
pw_types = {pw_files[0]:'biocarta', pw_files[1]:'kegg', pw_files[2]:'canonical', pw_files[3]:'other'}
pathways = calcom.io.CCList()
pw_names = calcom.io.CCList()

for pwf in pw_files[:-1]: # Handle the last one separately; only want to look at influenza-related pathways there for now.
    with open(pw_prefix+pwf,'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for pw in reader:
            if pw[0] in pw_names:
                # skip; pathway is already accounted for (canonical seems to copy some if not all PWs from others)
                continue
            else:
                pw_names.append(pw[0])

                pathways.append(
                Pathway(
                name=pw[0],
                hyperlink=pw[1],
                genes=pw[2:],
                pw_type=pw_types[pwf]
                )
                )
            #
#

with open(pw_prefix+pw_files[-1],'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for pw in reader:
        # Does the gene set have any relevance for us?
        if re.match('[a-z0-9\_]*flu|h1n1|h3n2|rsv|hrv|rhino', pw[0].lower()):
            pw_names.append(pw[0])

            pathways.append(
            Pathway(
            name=pw[0],
            hyperlink=pw[1],
            genes=pw[2:],
            pw_type=pw_types[pw_files[-1]]
            )
            )
        #
#

def append_pathways_to_ccd(input_ccd):
    '''
    Takes all the pathways constructed above and appends them as
    feature sets in a calcom dataset.
    NOTE that this is a quick implementation; something more elaborate
    is in the works.
    '''
    try:
        vnames = np.array([str(s) for s in input_ccd.variable_names])
    except:
        vnames = np.array([s.decode('utf-8') for s in input_ccd.variable_names])
    #

    for pw in pathways:
        fset = calcom.io.CCList()
        for g in pw.genes:
            i = np.where(g==vnames)[0]
            if len(i)>0:
                fset.append(i[0])
        #
        if len(fset)>0:
            input_ccd.add_feature_set(pw.name,fset)
        #
    #
#
