.. calcom documentation master file, created by
   sphinx-quickstart on Fri May 29 10:58:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to calcom's documentation!
==================================

Calcom is a package written by members of the 
Pattern Analysis Lab at Colorado State University. 
There are several purposes to this package:

- Provide convenient interfaces to clustering and classification tools studied and developed within our research group, loosely following ".fit()" and ".predict()" syntax.
- Provide a data structure that can handle both "tidy" and "untidy" experimental data sets, including slicing, basic querying, data summarization and exploration, etc. In particular, we provide tools to "reference by name" for complex cross-validation and feature selection.
- Provide seamless wrappers to GPU functionality utilizing PyTorch. These interfaces currently exist for our Sparse Support Vector Machine (SSVM) and Centroidencoder tools.

Much of our recent work (at least 2017 onwards) has been focused on machine learning applied to biological data sets (gene expression, or other "omics" data sets). To that end, the "biomodules" folder includes some auxilliary tools related to working with gene expression data and pathways. These are built to work with our local computer cluster at CSU, and **you will need to modify these files yourself** if you wish to use them elsewhere.


Calcom package structure
==================================

.. toctree::
    :maxdepth: 3
    :caption: Primary package structure (visit links for member classes and functions):
   
    calcom

Example scripts
==================================
.. toctree::
    :maxdepth: 2
    :caption: A loose collection of example scripts showcasing various aspects of the package. A work in progress.
    
    examples

Miscellaneous biological modules
==================================
.. toctree::
    :maxdepth: 3
    :caption: A loose collection of modules for querying both online and local databases to access information about genes, and perform some conversions between naming conventions. Not comprehensive.

    biomodules

Sanity tests
==================================
.. toctree::
    :maxdepth: 2
    :caption: A loose collection of tests to sanity check the functionality of increasingly complex tasks within calcom. Not comprehensive.
    
    sanity

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

