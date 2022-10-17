import csv
# TODO: make this less hacky.
import os
prefix = os.path.abspath('.')
fname = prefix + '/animals.txt'

with open(fname, 'r') as f:
    csvr = csv.reader(f)
    animals = [ ','.join(row) for row in csvr ]
#
