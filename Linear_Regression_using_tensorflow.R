# Creating a Simple Linear Regression model using Tensorflow/Keras

# load keras and tensorflow
library(keras)
library(tensorflow)

# define model structure
# Keras has 2 programming interfaces APIs for creating models: sequential(which is why we use keras_model_sequential()) and functional API interfaces
mymodel <- keras_model_sequential() %>%
  layer_dense(units = 1, input_shape = 1,    # units = 1 is d Output layer with 1 neuron/node. input_shape = 1 means we are using just 1 independent column/variable(since it is a simple linear regression model)
              activation = "linear")


# compile model
mymodel |> compile(loss = "mse",
    optimizer = optimizer_adam(lr = 0.01),  # lr is learning rate
    metrics = list("mean_absolute_error"))

summary(mymodel)


# import data file
library(tidyverse)

mydata <- read_csv("Auto.csv")
str(mydata)

# select independent variable (x) and dependent variable (y) and convert to matrix becuz Tensorflow only works with vectors, matrix or arrays
x_train <- as.matrix(select(mydata, displacement))  # using only displacement column as independent variable since it is a simple linear regression model
y_train <- as.matrix(select(mydata, mpg))   # mpg column is response variable


# Train Linear model
ourdata <- mymodel |> fit(x = x_train, y = y_train,
                          epochs = 500, validation_split = 0,   # using validation_split = 0 means u wanto train d entire dataset
                          verbose = 1)  # performs 500 epochs/iterations

# print model weights (regression coefficients slope and bias)
mymodel$weights  # take note of d value inside numpy=array([[]])


#-------------------

# using diff values in compile and training of d linear model
# compile model
mymodel |> compile(loss = "mse",
                   optimizer = optimizer_adam(lr = 0.001),  # we change lr value
                   metrics = list("mean_absolute_error"))

summary(mymodel)


# Train Linear model
ourdata <- mymodel |> fit(x = x_train, y = y_train,
                          epochs = 500, validation_split = 0,   # we change verbose value
                          verbose = 0)

# print model weights (regression coefficients slope and bias)
mymodel$weights    # take note of d value inside numpy=array([[]])

#--------------

# predict on the training set
mydata$predicted_tf <- predict(mymodel, x_train)  # creates a new column- predicted_tf in d main dataset with predicted values
head(mydata)     # u can compare predicted_tf column(which is d predicted values) and mpg column




# Lets create Simple Linear regression using d normal Linear regression function 
# and compare its value with d value from using Tensorflow on linear regression just done above
lm_fit <- lm(mpg ~ displacement, data = mydata)
lm_fit          # d values of d imdependent variable here should be close to the value inside numpy=array([[]]) in mymodel$weights noted above

mydata$predicted_lm <- predict(mymodel, x_train)  # creates a new column- predicted_tf in d main dataset with predicted values
head(mydata)    # and we compare predicted_tf column and predicted_lm column and see that they are d same or almost d same


# find regression coefficients with base R optimization
# using a function to optimize d RMSE
fr <- function(x) {   
  x1 <- x[1]         # intercept or bias
  x2 <- x[2]          # slope of coefficient of d independent variable
  mean((y_train - (x1 + x2 * x_train))^2)   # y_train - (x1 + x2 * x_train)  is actual values - predicted values(of our model)
}
optim(par = c(0, 0), fn = fr)   # c(0, 0) means x1(intercept) = 0 and x2(slope) = 0 
# values in $par is almost same with the values in lm_fit object above(which is d linear model)
?optim