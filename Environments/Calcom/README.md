## What is calcom?
Calcom ("**Cal**culate and **Com**pare") is a Python package that we have developed to
disseminate the many algorithms developed at the Pattern
Analysis Laboratory at Colorado State University for machine learning,
data visualization, in an easy-to-use format. We also
are developing data structures to handle heterogeneous/multiple
datatypes, making basic queries, and so on, to aid in
complex classification and cross-validation schemes.

Broadly, the intention of the package is to:
* Make it as simple as possible for an end user
to try our algorithms on their data;
* Make it as simple as possible for someone who wishes to
implement their own algorithms for preprocessing, classification,
visualization, etc, in the package, only requiring they fit a template
with standardized inputs and outputs;
* Make it simple to compare multiple algorithms on a single dataset,
and share the relevant algorithms/parameters with others.

Calcom is implemented as a Python 3 package which is
installable locally via the pip package manager. There are plans
to implement a GUI in the future, but not until the core functionality
and structure of the backend has stabilized.

## Installation; dependencies
The package is developed and supported only in Python 3.
Note that a standard installation using the pip package manager
will automatically install dependencies listed in the setup.py file
in the main directory. We have not done any testing using
other distributions, e.g. Anaconda; we'd be happy to hear from users
if standard installation mantras succeed or not in those distributions.

This is the most recent list of dependencies. Using a package
manager will automatically verify these are satisfied; otherwise they
will be automatically downloaded and installed.
1. Click >= 6.0
2. numpy >= 1.8.2
3. scipy >= 1.3.0
4. sklearn >= 0.21.3
5. matplotlib >= 3.0.0
6. pandas >= 0.25.0
7. h5py >= 2.10.0

Optional tools associated with our research projects may require
fetching and parsing information from online databases, working with
xlsx files, etc. The packages below are for these purposes, and
will *not* be installed. Install as needed.
* xlrd
* lxml
* requests

Optional dependencies:
1. torchvision, for GPU acceleration in specific algorithms
(in particular Centroidencoder and the torch implementation of
Sparse Support Vector Machines).

### If you are on the Katrina cluster:
Calcom is already installed globally on Katrina on the latest
version of Python 3 for each respective machine. Unless you have a
very good reason, you probably should *not* install your own copy.

### If you are on a personal computer:

1. Clone the repository via `git clone` or otherwise download the source code.
2. Navigate to the main directory. This directory should have a setup.py file.
3. Use python3's pip to install with the desired settings. We highly recommend
the following approach:
    * python3 -m pip install -e .
    * If you do not have root privileges, or wish to install locally
    anyways, include the "--user" flag at the end of the above incantation.
    Note you may need to modify your PYTHONPATH to seamlessly
    import calcom in a script if you do this.
    * Alternatively, set up the package in a virtual environment.

## Example usage
This demonstrates a very simple use case of building a classifier
on randomly generated data with labels.
More realistic examples on real datasets will be provided in the near future.
Observe that calls to the classifier obey the same ".fit()", ".predict()" scheme as
those found in scikit-learn, allowing for simplified comparisons (i.e., looping over
classifiers effortlessly).

```
import calcom
import numpy as np

n = 100     # number of datapoints
d = 4       # dimensionality of data

data = np.random.randn(n,d)
labels = np.random.choice( ['lion', 'tiger'], n)    # random binary labels

# Initialize a classifier -- this one randomly assigns labels.
rc = calcom.classifiers.RandomClassifier()

# Fit on the first half of the data, then predict on the
# second half of the data.

rc.fit( data[ :n//2 ], labels[ :n//2 ] )
predicted = rc.predict( data[ n//2: ] )

# Evaluate the average of success rates on each class ("balanced success rate")
bsr = calcom.metrics.ConfusionMatrix('bsr')

print('BSR: ', bsr.evaluate(labels[n//2:], predicted))
```
