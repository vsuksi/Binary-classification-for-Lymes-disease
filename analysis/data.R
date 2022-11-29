library(readxl)
library(dplyr)

# For pipeline development
dataset <- "USA data"
tse <- read_usadata()

# For our own testing
# "220810_Compound_Discoverer_Metabo_Lyme_May_2021_MS1_5E6_IK_modified.xlsx"
#f <- "data/mzMine_report_MetaboLyme_May_2021_modified_IK.xlsx"
#dataset <- f
#tse <- read_turkudata(f)
