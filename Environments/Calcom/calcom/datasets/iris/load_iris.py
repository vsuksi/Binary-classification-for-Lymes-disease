import csv
import numpy as np
import calcom
import os

f_data = 'iris.data' # note - relative path.
f_info = 'iris.names'
parent_dir = os.path.dirname(__file__)

with open(os.path.abspath(parent_dir + '/' + f_data), 'r') as f:
    csvr = csv.reader(f)
    rawdata = list(csvr)
#

# last row is empty
rawdata = rawdata[:-1]

data = np.array( [row[:-1] for row in rawdata], dtype=float )
labels = [['flower_type']] + [[row[-1]] for row in rawdata]
variable_names = ['sepal_length', 'sepal width', 'petal length', 'petal width']

ccd = calcom.io.CCDataSet()
ccd.create(data, labels, variable_names=variable_names)

# add the description, why not
with open(os.path.abspath(parent_dir + '/' + f_info), 'r') as f:
    raw_read = f.readlines()
ccd.add_about(''.join( raw_read ), add_autosummary=False)
