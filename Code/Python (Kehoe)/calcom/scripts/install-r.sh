# This script is only for linux; Tested on Ubuntu
# if you are using a virtualenv for the project, please make sure to run the script inside virtualenv 
# https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-16-04-2
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
sudo apt-get update
sudo apt-get install r-base

pip install rpy2
# install limma library in R
# give permission to add library
sudo chmod o+w /usr/local/lib/R/site-library
R < install-limma.R --no-save
