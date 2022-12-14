
extract_coldata <- function (tab) {
  require(S4Vectors)
  coldata <- tab[1:4, -c(1,2,3)]
  coldata <- DataFrame(t(coldata))
  colnames(coldata) <- unlist(tab[1:4,3], use.names=FALSE)
  rownames(coldata) <- NULL
  coldata$Group <- factor(coldata[, "Group"])
  coldata$Subject <- factor(coldata[, "Subject"])
  coldata$Time <- as.numeric(coldata[,"Time (months)"])
  coldata[, "Time (months)"] <- NULL
  coldata
}

extract_rowdata <- function (tab) {
  require(S4Vectors)
  require(lubridate)
  rowdata <- DataFrame(tab[-c(1:5), 1:3])
  colnames(rowdata) <- as.character(unlist(tab[5,1:3]))
  rowdata[, "Rt"] <- rowdata[, "Rt (min)"]
  rowdata[, "Rt (min)"] <- NULL
  rowdata[, "CompoundID"] <- rowdata[, "Compoud no."]
  rowdata[, "Compoud no."] <- NULL
  # Set types
  rowdata[,1] <- as.numeric(rowdata[,1])
  rowdata[,2] <- as.numeric(rowdata[,2])
  rowdata[,3] <- as.numeric(rowdata[,3])
  rowdata
}

extract_assaydata <- function (tab) {
  x <- apply(tab[-seq(5), -seq(3)], 2, as.numeric)
  rownames(x) <- NULL
  colnames(x) <- NULL
  x
}





read_usadata <- function () {

  #We load data with already inputted missing values using kNN algorithm
  #(that's why in the pipeline method we comment that part, as it takes
  #time to execute).


  df <- read.csv("../dataset.csv") %>% select(-X)
  tse <- TreeSummarizedExperiment(
         colData = DataFrame(Sample=paste0("S",as.character(seq(nrow(df)))), Group=df$outcome),
	 assay = SimpleList(signal=t(as.matrix(df[, 1:4851])))
       )
  tse
}


read_turkudata <- function (f) {

  # Read the source file
  tab <- read_excel(f, col_names=FALSE)

  # Extract data components
  coldata <- extract_coldata(tab)
  rowdata <- extract_rowdata(tab)
  assaydata <- extract_assaydata(tab)

  # Validate component match
  if (!nrow(assaydata)==nrow(rowdata)) {stop("")}
  if (!ncol(assaydata)==nrow(coldata)) {stop("")}

  # Create TreeSummarizedExperiment
  tse <- TreeSummarizedExperiment(
         colData = coldata,
         rowData = rowdata,
	 assay = SimpleList(signal=assaydata)
       )

  tse

}
