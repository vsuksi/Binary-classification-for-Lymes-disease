#All is selected apart from the outcome variable

x <- data % >%
select ( - outcome )

y <- data [ , " outcome " ]


#results of feature selection are saved in "result", which is ordered in "features"

disp _ features <- function ( result , number ) {
20 features <- result $ fs . order [ number ]
