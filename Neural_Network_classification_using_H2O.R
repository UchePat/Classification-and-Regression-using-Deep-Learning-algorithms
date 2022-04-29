# To use H2O package for Neural Network classification, the response/dependent variable in d dataset should be either a numeric variable or a categorical factor variable

# Load packages
library(h2o)
library(mlbench)

# Explore dataset
data("BreastCancer")  # dis dataset is in mlbench package

# Remove ID column 
mydata <- BreastCancer[, -1]  # removes d 1st column(which is an ID column) from d dataset
str(mydata)     # the response/dependent variable is class and it is a categorical factor variable

# changing all d ordered factors(which have numeric values) columns to numeric columns/variables
mydata$Cl.thickness <- as.numeric(mydata$Cl.thickness)
mydata$Cell.size <- as.numeric(mydata$Cell.size)
mydata$Cell.shape <- as.numeric(mydata$Cell.shape)
mydata$Marg.adhesion <- as.numeric(mydata$Marg.adhesion)
mydata$Epith.c.size <- as.numeric(mydata$Epith.c.size)

str(mydata) 


# Begin H2O instance. You'll need Java installed (Java versions supported by h2o: 8-13)
h2o.init()


# Convert input/dataset into H2O Frame. You need to be online to run d code below 
mymodel <- as.h2o(mydata)


# In practice, you will want to split d dataset into two: a training set and a test set, 
# den you will train the model using the train data, and evaluate model performance on the test set. 
# but in dis project we are not going to split d dataset, we are going to use d whole dataset


# Take a look at the contents of the H2OFrame
h2o.describe(mymodel)   # since H2O runs on Java, u can see that dat Type column values uses Java syntax for variable datatypes


# Create neural network using H2O
?h2o.deeplearning

mynet <- h2o.deeplearning(x = 1:9,          # x is d independent variables/columns
                           y = 10,          # y is d dependent variable/column
                           training_frame = mymodel,
                           nfolds = 5,            # performing 5 fold cross-validation
                           standardize = TRUE,
                           activation = "Rectifier",
                           hidden = c(5, 200),      # dis gives 2 Hidden layers: 1st Hidden layer has 5 neurons/nodes and d 2nd Hidden layer has 200 neurons/nodes. the default is c(200,200)
                           seed = 2021)

# Look at performance of model
h2o.performance(mynet)   # displays confusion matrix etc. we view d accuracy of d model in max accuracy under value column


# Use model to predict class column values
mypred <- h2o.predict(object = mynet, 
                       newdata = mymodel)   # since we did not split d dataset, we will predict d entire dataset
head(mypred)

# Explore predictions
mypred <- as.data.frame(mypred)
head(mypred)

# Shutdown H2O instance
h2o.shutdown()
