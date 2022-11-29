#import h5py    #switched to a lazy import
import re
from urllib import request
#from calcom.utils import type_functions as tf
import time
import numpy as np


# For the record: the API to get all gene ID from NCBI (max 100000 at a time)
# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=id&retstart=100&retmax=100000

class GeneInfo:
    '''
    A class to get at gene information from bioDBnet using their
    API, and longer descriptions from NCBI directly parsing the
    corresponding HTML page (can we do this better?)
    '''
    def __init__(self,gid,**kwargs):
        self.species = kwargs.get('species',9606)   # human
        self.format = kwargs.get('format','geneid') # others: ensembl, illumina
        self.verbosity = kwargs.get('verbosity',0)
        self.pause_time = kwargs.get('pause_time',0.0)  # seconds to pause after reading to allow bioDBnet's ancient server to catch up

        self.identifiers = ['geneid', 'illuminaid', 'affyid']    # hard-coded for now
        self.identifier_tags = {'geneid':'GeneID', 'illuminaid':'IlluminaID', 'affyid':'AffyID'}
        self.version = "1.0"
        for identifier in self.identifiers:
            setattr(self, identifier, '')
        # self.geneid = ''
        # self.illuminaid = ''
        # self.affyid = ''
        self.geneid_arr = []

#        setattr(self, self.format, gid.decode('utf-8') if tf.is_bytes_like(gid) else gid)

        # calcom dependency too heavy for its limited use here.
        bytes_likes = [bytes,np.bytes_]
        setattr(self, self.format, gid.decode('utf-8') if (type(gid) in bytes_likes) else gid)
        # self.geneid = gid.decode('utf-8') if tf.is_bytes_like(gid) else gid

        self.gene_symbol = ''
        self.description = ''
        self.long_description = ''
        self.genetic_source = ''
        self.other_designations = []
        self.scientific_name = ''
        self.common_name = ''

        self.nomenclature_name = ''

        # self.gsym_pat = ".*\<GeneSymbol\>([a-zA-Z0-9\-\:\.\,\&\'\(\)\/\ ]*)\<"  # Gene symbol pattern
        self.desc_pat = ".*\[Description: (.*?)\].*"     # Gene description pattern
        # self.kegg_pat = ".*\<KEGGPathwayTitle\>([a-zA-Z0-9\-\:\.\,\&\'\(\)\/\ ]*)\<"
        # self.reac_pat = ".*\<ReactomePathwayName\>([a-zA-Z0-9\-\:\.\,\&\'\(\)\/\ ]*)\<"

        # TODO: is there a faster way to access this data?
        # The NCBI pages aren't very compact.
        #self.ncbi_pat = '.*\<dt\>Summary\<\/dt\>[\ ]*\<dd\>([a-zA-Z0-9\-\ \(\)\'\,\.\;\%\[\]\:\/\&]*)'

        self.kegg_pathways = []
        self.reactome_pathways = []

        # TODO: any extra information.
        #       Option to download more junk.
        #       Parsing the gene info further (extend self.desc_pat).
        #       Working with other gene formats (EnsemblID, IlluminaID)
        #       Handling empty KEGG/REACTOME pathways properly; these come with a single dash "-".
        #       BIOCARTA PATHWAYS - note this is a bit more involved, since the formatting
        #           is different. See for example:
        #           https://biodbnet-abcc.ncifcrf.gov/webServices/rest.php/biodbnetRestApi.xml?method=db2db&format=row&input=geneid&inputValues=1234&outputs=genesymbol,geneinfo,reactomepathwayname,keggpathwaytitle,biocartapathwayname&taxonId=9606
        return
    #

    def generate_biodbnet_url(self):
        gfmt = self.format.lower()

        prefix = 'https://biodbnet-abcc.ncifcrf.gov/webServices/rest.php/biodbnetRestApi.xml?method=db2db&format=row'
        inputs = '&input=%s&inputValues=%s' % (gfmt, getattr(self, self.format))    # The input gene's identifier
        suffix = '&outputs=genesymbol,geneinfo,reactomepathwayname,keggpathwaytitle,geneid,illuminaid,affyid'
        misc = '&taxonId=%i'%self.species

        url = prefix + inputs + suffix + misc
        return url
    #

    def generate_ncbi_url(self):
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id=" + self.geneid
        return url
    #


    def download_page(self,url):


        import requests
        verb = self.verbosity
        gid = self.geneid


        not_downloaded = True
        maxtries = 5
        ntries = 0
        response = None
        while not_downloaded and ntries<=maxtries:
            try:
                response = requests.get(url)
                not_downloaded = False
            except:
                ntries += 1
                if verb>0:
                    print(url)
                    print('url for %s failed to download %i times. Retrying.'%(gid,ntries))
                time.sleep(self.pause_time)
            #
        #

        return response
        #
    #

    def parse_biodbnet(self):
        import lxml.objectify

        verb = self.verbosity
        res = self.download_page(self.generate_biodbnet_url())
        if not res:
            return
        tree = lxml.objectify.fromstring(res.content)

        # Getting gene description.
        try:
            #print(tree.item.GeneInfo)
            match = re.match(self.desc_pat, str(tree.item.GeneInfo))

            descr = match.groups(1)[0]
            vdesc = descr
        except:
            if verb>0:
                print('Failed to match a description for GeneID %s. Continuing.'%self.geneid)
            vdesc = ''
        #

        # Getting other gene identifiers.
        for identifier in self.identifiers:
            if identifier != self.format:
                try:
                    tag = self.identifier_tags[identifier]
                    value = getattr(tree.item, tag)
                    setattr(self, identifier, str(value))
                    if identifier == 'geneid':
                        self.geneid_arr = self.geneid.split('//')
                except:
                    continue
                #
            #
        #

        # Getting gene symbol.
        try:
            gsym = str(tree.item.GeneSymbol)
        except:
            if verb>0:
                print('Failed to match a gene symbol for GeneID %s. Continuing.'%self.geneid)
            gsym = ''
        #

        # Getting KEGG pathway info.
        try:
            raw_result = str(tree.item.KEGGPathwayTitle)
            try:
                kegg_result = raw_result.split('//')
            except:
                kegg_result = raw_result
            #
        except:
            if verb>0:
                print('Failed to match KEGG pathways for GeneID %s. Continuing.'%self.geneid)
            kegg_result = []
        #


        # Getting Reactome pathway info.
        try:
            raw_result = str(tree.item.ReactomePathwayName)
            try:
                reac_result = raw_result.split('//')
            except:
                reac_result = raw_result
            #
        except:
            if verb>0:
                print('Failed to match Reactome pathways for GeneID %s. Continuing.'%self.geneid)
            reac_result = []
        #

        # TODO: HANDLE THE "-" RETURN WHEN THERE ARE NO PATHWAYS CORRESPONDING TO
        # THE GENE.
        self.description = vdesc
        self.gene_symbol = gsym
        self.kegg_pathways = kegg_result
        self.reactome_pathways = reac_result

        return
    #

    def parse_ncbi(self,**kwargs):
        '''
        Since the NCBI pages are so large, don't bother storing them.
        Just pull what we want and dump the rest.
        '''

        verb = kwargs.get('verbosity',0)

        if len(self.geneid) < 2:
            if self.verbosity:
                print("Need Gene ID to fetch data from NCBI")
            return

        import lxml.objectify

        res = self.download_page(self.generate_ncbi_url())
        if not res:
            return
        tree = lxml.objectify.fromstring(res.content)
        doc = tree.DocumentSummarySet.DocumentSummary
        if verb>0:
            print(tree)
            print(doc)
        #print(tree.DocumentSummarySet.DocumentSummary.Description)
        self.long_description = str(doc.Summary)
        self.genetic_source = str(doc.GeneticSource)
        self.other_designations = str(doc.OtherDesignations).split("|")
        self.scientific_name = str(doc.Organism.ScientificName)
        self.common_name = str(doc.Organism.CommonName)

        self.nomenclature_name = str(doc.NomenclatureName)

        desc = str(doc.Description)
        # keep the longer description between biodbnet and NCBI
        if len(desc) > len(self.description):
            self.description = desc
        return
        #
        if verb>0:
            return doc
    #

    def write_to_file(self,filename, **kwargs):
        # with open("illumina_to_geneid.txt", "a") as myfile:
        #     myfile.write(self.illuminaid + "\t" + ';'.join(self.geneid_arr) + '\n')
        import h5py
        compression_level = kwargs.get('compression_level',0)

        if self.format == 'geneid':
            with h5py.File(filename, 'a') as f:
                f.attrs['version'] = self.version
                if "/GeneID" not in f:
                    h5_Gene = f.create_group('GeneID')
                else:
                    h5_Gene = f['GeneID']
                if "/GeneID/" + self.geneid not in f:
                    dset = h5_Gene.create_dataset(self.geneid,data=[], compression="gzip", compression_opts=compression_level)
                    dset.attrs['illuminaid'] = self.illuminaid
                    dset.attrs['description'] = self.description
                    dset.attrs['gene_symbol'] = self.gene_symbol
                    dset.attrs['kegg_pathways'] = ";".join(self.kegg_pathways)
                    dset.attrs['reactome_pathways'] = ";".join(self.reactome_pathways)

                    #ncbi data
                    dset.attrs['long_description'] = self.long_description
                    dset.attrs['genetic_source'] = self.genetic_source
                    dset.attrs['other_designations'] = ";".join(self.other_designations)
                    dset.attrs['genetic_source'] = self.genetic_source
                    dset.attrs['scientific_name'] = self.scientific_name
                    dset.attrs['common_name'] = self.common_name

            #
        #
        elif self.format == 'illuminaid':
            with h5py.File(filename, 'a') as f:
                f.attrs['version'] = self.version
                if "/IlluminaID" not in f:
                    h5_Illumina = f.create_group('IlluminaID')
                else:
                    h5_Illumina = f['IlluminaID']
                if "/IlluminaID/" + self.illuminaid not in f:
                    dset = h5_Illumina.create_dataset(self.illuminaid,data=[], compression="gzip", compression_opts=compression_level)
                    dset.attrs['geneid'] = self.geneid
                    dset.attrs['description'] = self.description
                    dset.attrs['gene_symbol'] = self.gene_symbol
                    dset.attrs['kegg_pathways'] = ";".join(self.kegg_pathways)
                    dset.attrs['reactome_pathways'] = ";".join(self.reactome_pathways)

            #
        #


    def fetch(self):
        self.parse_biodbnet() #   TEMPORARY
        #time.sleep(self.pause_time)
        if self.format == 'geneid':
            self.parse_ncbi()

        return
    #

    def summarize(self):
        print('Summary for %s %s - %s\n'%(self.format, getattr(self, self.format), self.description))
        print('Gene ID: %s\n' % self.geneid_arr)
        print('KEGG Pathways:')
        for kp in self.kegg_pathways:
            print('\t%s'%kp)
        print('\nReactome Pathways:')
        for rp in self.reactome_pathways:
            print('\t%s'%rp)
        print('')


        print('NCBI gene description:\n')
        print('\t%s'%self.long_description)
        print('\nGenetic Source: %s\n'%self.genetic_source)
        print('\nOther Designations:')
        for od in self.other_designations:
            print('\t%s'%od)
        print('\n=========================================')

        return
    #
#
import concurrent.futures
executor = None

def illumina_thread():
    with open('Illumina_ID__to__Gene_ID_9606.txt', 'r') as f:
        lines = f.readlines()

        def worker(l):
            ilmn_id = l.split()[0]
            print(ilmn_id)
            g1 = GeneInfo(ilmn_id, format='illuminaid')
            g1.verbosity = 1
            g1.fetch()
            g1.write_to_file("Genedb.h5")
            g1.summarize()

        for l in lines:
            executor.submit(worker, l)
            #worker(l)

def gene_thread(**kwargs):
    # with open('Gene_ID__to__Ensembl_Gene_ID_9606.txt', 'r') as f:
    #     lines = f.readlines()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers = 10)

    gids = kwargs.get('gids',[])
        # def worker(l):
        #     geneid = l.split()[0]
        #     print(geneid)
        #     g1 = GeneInfo(geneid)
        #     g1.verbosity = 1
        #     g1.fetch()
        #     g1.write_to_file("Genedb.h5")
        #     g1.summarize()
    # results = []
    def worker(gid):
        # j,myg = a
        # geneid = l.split()[0]
        geneid = gid
        print(geneid)
        g1 = GeneInfo(geneid)
        g1.verbosity = 1
        g1.fetch()
        g1.write_to_file("Genedb_nuch.h5")
        g1.summarize()
        # results.append([j,g1])
    #

    # for l in lines:
    for j,myg in enumerate(gids):
        executor.submit(worker, myg)
        #worker(l)
    # return results
#


if __name__=="__main__":
    '''
    Example usage of the class.
    '''
    executor = concurrent.futures.ThreadPoolExecutor(max_workers = 10)
    import calcom
    ccd = calcom.io.CCDataSet('../../ccd_gse73072_geneid.h5')
    glist = list(ccd.variable_names)
    glist = glist
    # gene_thread()
    gene_thread(gids=glist)
    illumina_thread()

    # import threading
    # il = threading.Thread(target=illumina_thread)
    # ge = threading.Thread(target=gene_thread)
    # il.start()
    # ge.start()

#
