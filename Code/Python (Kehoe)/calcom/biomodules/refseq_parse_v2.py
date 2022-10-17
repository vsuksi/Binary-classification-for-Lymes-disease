class refseq_parse:

    def __init__(self):
        from Bio import Entrez
        import numpy
        # import concurrent.futures
        from multiprocessing import Manager

        self.Entrez = Entrez
        self.np = numpy
        self.email = ''
        self.manager = Manager()
        self.results = self.manager.dict()

        self.Entrez.email = self.email

        return
    #

    def refseq_parse(self, refseq_id, verbosity=0, return_string=True, finish_msg=None):

        #Entrez.email = email
        handle = self.Entrez.efetch(db="nucleotide", id=refseq_id,  retmode="xml")
        records = self.Entrez.parse(handle)

        # On the grand scheme of things, we're looking for any mention
        # of 'GeneID' in these records. We have reason to believe that
        # they'll be located in a certain location in the tree.
        # For the moment go with that working hypothesis and see what happens.
        #
        # Probably just storing the entire result is the easiest thing;
        # who knows if this can be accelerated if we know better how to
        # ask for what we want.
        mo = []
        for r in records: mo.append(r)


        if False:
            gbseq_quals = mo[0]['GBSeq_feature-table'][2]['GBFeature_quals']

            entrez_ids = []
            for j,entry in enumerate(gbseq_quals):
                if verbosity>0:
                    print( 'Searching record %i of %i...'%(j+1,len(gbseq_quals)) , end='\r')
                magic_phrase = 'GBQualifier_value'
                if magic_phrase in entry.keys():
                    candidate = entry[magic_phrase]
                    l = candidate.split(':')
                    if l[0]=='GeneID':  # we have a winner
                        entrez_id = l[1]
                        if verbosity>0:
                            print('\nMatch found: %s <-> %s'%(refseq_id, entrez_id))
                        entrez_ids.append( entrez_id )
                else:
                    if verbosity>0:
                        print('',end='\b')
            #
        else:
            # Looking in the wrong place?
            import re
            entrez_ids = re.findall('GeneID[\ \:]{0,}([0-9]{1,})', str(mo))
        #

        entrez_ids = self.np.unique(entrez_ids)
        if len(entrez_ids)==1:
            entrez_ids = entrez_ids[0]
        elif len(entrez_ids)==0:
            if verbosity>0:
                print('No match found for %s.'%refseq_id)
            entrez_ids = ''

        # Add to class's dictionary.
        self.results[refseq_id] = entrez_ids

        if finish_msg:
            print(finish_msg)

        if return_string:
            return entrez_ids
        else:
            pass
    #

    def refseq_batch_parse(self, refseq_ids, verbosity=0, max_workers=16):
        '''
        Concurrent calls of refseq_parse using concurrent.futures.ThreadPoolExecutor().
        Results stored in self.results.
        '''
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        if verbosity>0:
            for j,r in enumerate(refseq_ids):
                fm = '%s  (%i of %i)'%(r,j+1,len(refseq_ids))
                executor.submit(self.refseq_parse, r, return_string=False, finish_msg=fm)
            #
        else:
            for j,r in enumerate(refseq_ids):
                executor.submit(self.refseq_parse, r, return_string=False)
            #
        return
    #

#

