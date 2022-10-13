df <- read.csv("PeakTable_data.csv") # 4851-dimensional feature vectors
df1 <- read.csv(" HCvsEL_clinicaldata.csv" ) # Diagnoses: EDL, ELL, HCN, HCE1


# sub(pattern, replacement, x)
# gsub(pattern, replacement, x)
# pattern: synonym to string
# sub replaces the first occurrence while gsub replaces all occurrences

# Deleting dates from the column names (after the dot):
colnames (df) <- gsub ( " \\.. * " ," " , colnames ( df ) )


# transpose(l, fill, ignore.empty, keep.names, make.names)
# l: a list, data.frame or data.table
# fill: fill shorter elements so the transposed elements are equal lengths
# ignore.empty: ignore elements with length 0
# keep.names: assign name to the column in results containing names of input
# make.names: choose name or number of column in input to use as output names
# Transposing (rows to columns)
dff <- transpose (df, keep.names = " rn " )
