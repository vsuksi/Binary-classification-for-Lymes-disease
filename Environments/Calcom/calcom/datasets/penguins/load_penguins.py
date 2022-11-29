import calcom
import os
#import csv
import pandas
import numpy as np

parent_dir = os.path.dirname(__file__)

f_raw = 'penguins_raw.csv'
f_raw_full = os.path.abspath(parent_dir + '/' + f_raw)

if not os.path.exists(f_raw_full):
    raise Exception('File %s does not exist; penguin data cannot be loaded.'%f_raw_full)

df = pandas.read_csv(f_raw_full)

data_cols = [
    'Culmen Length (mm)', 
    'Culmen Depth (mm)', 
    'Flipper Length (mm)', 
    'Body Mass (g)'
]

attr_cols = np.setdiff1d(list(df), data_cols)

data = df[data_cols].values
attrs = df[attr_cols]

ccd = calcom.io.CCDataSet()

ccd.create(
    data, 
    metadata=np.vstack([attr_cols,attrs]), 
    variable_names=data_cols
)

# just use the readme in the same file 
# to create the "about".
readme_path = parent_dir + '/readme.txt'
if os.path.exists(readme_path):
    with open(readme_path, 'r') as f:
        ccd.add_about(f.read(), add_autosummary=False)
else:
    ccd.add_about('''
~~~~~~~~~~~~~~~~~~~~~~
Penguins dataset
~~~~~~~~~~~~~~~~~~~~~~

This is a data set of penguins in Antarctica. 
In terms of machine learning, the goal is to 
classify a new penguin by its species given 
its biological characteristics. There is a 
lot of additional characteristics to these 
penguins that make for interesting extensions 
beyond a plain 3-class problem.
    ''',
    add_autosummary=False)
