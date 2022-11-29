
#from tables import *


def _gene_orthologs_csv_to_hdf(csvfile, h5file):
    '''
    Helper function to convert from csv to hdf
    Inputs:
        csvfile: name of the csv file
        h5file: name of the hdf5 file to be created
    Outputs:
        hdf5 file
    '''
    
    class GeneOrthologs(IsDescription):
        '''
        Utility class to save data into dataset
        '''
        from_tax = Int64Col()
        from_gene = Int64Col()
        to_tax = Int64Col()
        to_gene = Int64Col()

    h5file = open_file(h5file, mode="w", title="Gene Orthologs")
    table = h5file.create_table("/", 'data', GeneOrthologs, "Data")
    gene_orthologs = table.row
    with open(csvfile, "r") as f:
        #skip the first line
        f.readline()
        lines = f.readlines()
        for line in lines:
            arr = line.split()
            gene_orthologs['from_tax'] = int(arr[0])
            gene_orthologs['from_gene'] = int(arr[1])
            gene_orthologs['to_tax'] = int(arr[3]) 
            gene_orthologs['to_gene'] = int(arr[4])
            gene_orthologs.append()            
    table.flush()
    h5file.close()

def convert(filepath, from_tax, from_gene, to_tax):
    '''
    Convert `from_gene` from species `from_tax` to species `to_tax`
    Inputs:
        filepath: Gene Orthologs (HDF5) filepath
        from_tax: taxonomy of the given species
        from_gene: geneId of the given species
        to_tax: taxonomy of the species for which we are looking for a GeneID
    Outputs:
        to_gene: geneId of the requested species
    '''
    h5file = open_file(filepath, mode="r", title="Gene Orthologs")
    table = h5file.root.data

    to_gene = [x["to_gene"] for x in table.iterrows() if x['from_tax'] == from_tax and x['from_gene'] == from_gene and x['to_tax'] == to_tax]

    if len(to_gene) < 1:
        to_gene = [x["from_gene"] for x in table.iterrows() if x['to_tax'] == from_tax and x['to_gene'] == from_gene and x['from_tax'] == to_tax]
    h5file.close()
    return to_gene[0]


if __name__ == "__main__":
    # _gene_orthologs_csv_to_hdf("gene_orthologs.csv","gene_orthologs.h5")
    ret = convert("gene_orthologs.h5", 7994, 103025831 ,7955)
    print(ret)

