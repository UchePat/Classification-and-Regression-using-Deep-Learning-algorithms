# Self-Organizing Maps is also called Kohonen maps or Kohonen Artificial Neural Networks
# S.O.M are useful for clustering and data visualization.

# S.O.M is used to convert high dimensional input(ie where there are many variables/columns) into low dimensional map
# Nodes/circles closer to each other are similar and related to each other.

# install.packages("kohonen")
library(kohonen)


#--- Unsupervised Self-Organizing Maps ----
# Data
mydata <- read.csv(file.choose(), header = T)  # select binary.csv file
str(mydata)

X <- scale(mydata[,-1])  # removes d 1st column which is d dependent variable (and a categorical variable)
summary(X)


# creating S.O.M model
set.seed(222)

g <- somgrid(xdim = 4, ydim = 4, topo = "rectangular")

map <- som(X, grid = g, alpha = c(0.05, 0.01),   # alpha means learning rate
           radius = 1)
map
map$unit.classif  
head(mydata)   # comparing map$unit.classif and head(mydata), we can see that d 1st index number has small gre value and large gpa and rank values which is same with 11th node that it classified into
map$codes  # displays d probability values of d each size of each column in each node/circle

plot(map)      # displays how d size of each column vary in each node. we see that rank is largest in d 1st node at top row and 2nd node at bottom row   

plot(map, type = "changes")

plot(map, type = "codes", palette.name = rainbow,
     main = "4 by 4 Mapping of Application data")

plot(map, type = "count")  # displays a Heatmap showing how many datapoints are in each node- we see that 6th node has highest number of datapoints

plot(map, type = "mapping")  # displays a scatterplots showing how many datapoints are in each node

plot(map, type = "dist.neighbours")



#------------------------------------------------------------------



#--- Supervised Self-Organizing Maps ------
# Data
mydata <- read.csv(file.choose(), header = T)  # select binary.csv file
str(mydata)

# Data Split
set.seed(123)

ind <- sample(2, nrow(mydata), replace = T, prob = c(0.7, 0.3))

train <- mydata[ind == 1,]
test <- mydata[ind == 2,]


# Normalization of only numerical variables/columns
trainX <- scale(train[,-1])  # removes d 1st column which is d dependent variable (and a categorical variable)
trainX

testX <- scale(test[,-1], center = attr(trainX, "scaled:center"),  # we will normalize d testing dataset using mean and sd
               scale = attr(trainX, "scaled:scale"))
testX

trainY <- factor(train[,1])  # here we store d dependent variable which is d 1st column in training data

Y <- factor(test[,1])   # here we store d dependent variable which is d 1st column in testing data

test[,1] <- 0  # changes all d values in d dependent variable in testing data to 0

testXY <- list(independent_variables = testX, dependent_variable = test[,1])
testXY


# Classification & Prediction Model
set.seed(222)

map1 <- xyf(trainX, classvec2classmat(factor(trainY)),  # classvec2classmat - class vector to class matrix
            grid = somgrid(5, 5, "hexagonal"),
            rlen = 100)      # rlen is run length
plot(map1)

plot(map1, type = "changes")  # Matrix 1 is based on independent variable data, Matrix 2 is based on dependent variable data

plot(map1, type = "count")


# Prediction on dependent variable in Training data
mypred <- predict(map1)
mypred     

mypred1 <- predict(map1, newdata = testXY)
mypred1    


# Confusion Matrix
conf <- table(Predicted = mypred1$predictions[[2]], Actual = Y)  
conf

# Accuracy
sum(diag(conf)) / sum(conf)
#---- To improve d model accuracy, you can change d grid size values from 5, 5 to any values, change rlen value, change learning rate etc




# Displaying Cluster Boundaries
par(mfrow = c(1,2))

plot(map1, type = 'codes',
     main = c("Codes X", "Codes Y"))  # Codes X chart is for independent variable, Codes Y chart is for dependent variable

map1.hc <- cutree(hclust(dist(map1$codes[[2]])), 2)  # using Hierarchical Clustering

add.cluster.boundaries(map1, map1.hc)  # creates demarcating line in Codes Y chart thus separating d clusters

par(mfrow = c(1,1))
