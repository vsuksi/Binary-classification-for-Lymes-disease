library(magrittr)
library(dplyr)
library(mikropml)
library(data.table)

df <- read.csv(file = "train_data.csv")
df$.outcome <- as.factor(dfbinary$.outcome)
df_transposed <- transpose(df)
write.csv(df_transposed, "transposed_data.csv")
