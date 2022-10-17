#!/bin/sh

# Purpose: for future maintainers; the 
# incantation to make the documentation, 
# since we're not using out-of-the-box "make html" only.
# 
# Utilizing sphinx-apidoc with some options as well.
#
# ALSO build the latex, which can be compiled into a 
# pdf document later.

calcom_dir="../calcom/"
examples_dir="../examples/"
sanity_dir="../sanity/"
biomodules_dir="../biomodules/"

depth_level=5   # An upper bound for folder depth. Just want a big number.

# Reset source and copy in "master" files (homepage, configuration, CSS file)
rm -Rf ./source/*.rst
cp ./myconfig/index.rst ./source/index.rst
cp ./myconfig/conf.py ./source/conf.py
cp ./myconfig/calcom_theme.css ./source/_static/calcom_theme.css

#
# Compile rtfs for different folders separately using the
# sphinx-apidoc tool, which recursively searches through folders
# to build a full API documentation.
#
# See https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
# for details on parameters chosen here.
#

sphinx-apidoc -f -l --private -d ${depth_level} -o source/ ${calcom_dir}
sphinx-apidoc -f -l -d ${depth_level} -o source/ ${examples_dir}
sphinx-apidoc -f -l -d ${depth_level} -o source/ ${biomodules_dir}
sphinx-apidoc -f -l -d ${depth_level} -o source/ ${sanity_dir}

# compile html
make html

# compile latex
mkdir build_latex
sphinx-build -b latex source/ build_latex/
