df <- read.csv("PeakTable_data.csv") # 4851-dimensional feature vectors
df1 <- read.csv(" HCvsEL_clinicaldata.csv" ) # Diagnoses: EDL, ELL, HCN, HCE1

# Deleting dates from the column names (after the dot):
# sub(pattern, replacement, x)
# gsub(pattern, replacement, x)
# pattern: synonym to string
# sub replaces the first occurrence while gsub replaces all occurrences
colnames(df) <- gsub("\\..*" ," " ,colnames(df))


# Transposing (rows to columns):
# transpose(l, fill, ignore.empty, keep.names, make.names)
# l: a list, data.frame or data.table
# fill: fill shorter elements so the transposed elements are equal lengths
# ignore.empty: ignore elements with length 0
# keep.names: assign name to the column in results containing names of input
# make.names: choose name or number of column in input to use as output names
dff <- transpose (df, keep.names = " rn " )

# Adding an output column:
cols <- df1[,c("My.Number", "Sample.Type")]

# Merge by matching the two data frames to create an output column:
# Here the $-operator creates a new column
# match(x, table, nomatch, incomparables)
# x: the values to be matched
# table: the values to be matched against
# nomatch: assign value to be returned if no match is found
# incomparables: assign vector of values that musntn't be matched
dff$outcome_long <- cols$Sample.Type[match(dff$rn, cols$My.Number)]

# Delete first column with names
# subset(x, subset, select, drop = FALSE)
# subset: indicate elements or rows to keep
# select: indicate columns to select from dataframe
# drop: use `[` indexing operator
dff <- subset(dff, select = -c(rn))

# Change names of patient categories
# case_when() checks condition
# left side of "~" is the condition, right is output if true
dff$outcome_long <- case_when(dff$outcome_long == "Early Disseminated Lyme" ~   "EDL", dff$outcome_long == "Early Localized Lyme" ~ "ELL", dff$outcome_long == "Healthy Non -enemic control CO" ~ "HCN", dff$outcome_long == "Heatlhy Controls - Endemic Dr . Wormser" ~ "HCE1")

# Make the dependent variable categorical
dff$outcome _ long <- as.factor(dff$outcome_long)

# Indicate missing values
dff[dff== 0] <- NA

# Count missing values
sum(is.na(dff))

# Make new dataframe containing features for KNN imputation
df.feat <- dff[1:4851]

# Perform kNN with k = 5 and store results in new dataframe
res <- kNN(df.feat, k = 5)

# Make new dataframe for dependent variable
col.outcome <- dff[4852]

# Adding outcome column to imputed dataframe
res$outcome_long <- col.outcome

# Deleting unnecessary columns containing imputation metadata
# grep() searches for matches between argument and x
# grep(pattern, x, ignore.case, perl, value, fixed, useBytes, invert))
# pattern: string to be matched
# x: vector where matches are sought
# ignore.case: ignore assigned case
# perl: use Perl-compatible regexps
# value: return the values themselves
# fixed: match pattern string as is
# useBytes: match pattern byte by bytes
# invert: return indices or values that do not match
dataset <- res[grep("_imp ", colnames(res), invert = TRUE)]

# kNN imputation for missing values of dependent variable
dataset <- kNN(dataset, variable = "outcome_long")

# Temporary storage for dataset?
fin <- dataset[1:4852]

#The following should be zero if imputation was successful
sum (is.na(dataset))

# Make dependent variable a binary outcome
dataset$outcome_long <- case_when(fin$outcome_long == "EDL" ~ "lyme", fin$outcome_long == "ELL" ~ "lyme", fin$outcome_long == "HCN" ~ "healthy", fin$outcome_long == "HCE1" ~ "healthy")

#save as binary_outcome.csv
