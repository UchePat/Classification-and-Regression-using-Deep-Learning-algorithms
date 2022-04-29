# Creating a Logistical Regression model using Tensorflow/Keras

# load keras and tensorflow
library(keras)

# Define Model structure
# Keras has 2 programming interfaces APIs for creating models: sequential(which is why we use keras_model_sequential()) and functional API interfaces
mymodel <- keras_model_sequential() %>%
  layer_dense(units = 1, input_shape = 1,  # units = 1 is d Output layer with 1 neuron/node. input_shape = 1 means we are using just 1 independent column/variable
              activation = "sigmoid")


# compile model
mymodel |> compile(loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = 0.01),  # lr means learning rate
    metrics = list("accuracy"))
  
summary(mymodel)


# import data file
library(tidyverse)

mydata <- read_csv("cardio_CVDs.csv")
str(mydata)

# standardize d independent columns with numerical values(ie d continuous variables)
mydata$age <- scale(mydata$age)  # age column is d independent variable we are using so we standardize it becuz it has numerical values

# select predictor (x) and dependent variable (y) and convert to matrix becuz Tensorflow only works with vectors, matrix or arrays
x_train <- as.matrix(select(mydata, age))   # using only age column as independent variable 
y_train <- as.matrix(select(mydata, cardio))   # cardio column is response variable


# train Logistic model
ourmodel <- mymodel |> fit( x = x_train, y = y_train,
                          epochs = 100, validation_split = 0,   # using validation_split = 0 means u wanto train d entire dataset
                          verbose = 0)   

plot(ourmodel)

# print model weights (logistic regression coefficients)
mymodel$weights    # take note of d value inside d 2 numpy=array([[]])


# predict on the training set
mydata$predicted_tf <- predict(mymodel, x_train) # creates a new column- predicted_tf in d main dataset with predicted values
head(mydata)     # u can compare predicted_tf column(which is d predicted values) and age column(remember d age column has been standardized)

mydata$class_tf <- ifelse(mydata$predicted_tf < 0.5, 0, 1)  # creates a new column- class_tf in d main dataset based on stated values from predicted_tf column(ie if any value in predicted_tf column is < 0.5, write 0, else write 1)
head(mydata)   # compare class_tf column with age column


# examine confusion matrix of tensorflow model
library(caret)

confusionMatrix(as.factor(c(mydata$class_tf)), 
                as.factor(mydata$cardio), positive = '1')  # converting class_tf column and cardio column(d response variable) to factor datatype and comparing d values btw both columns using a confusion matrix
# we see that d accuracy value is too poor and there is too many misclassifications in d confusion  as such dis model can be improved




# Lets create Logistic regression using d normal Logistic regression function and compare its accuracy value with d accuracy value from using Tensorflow on Logistic regression just done above
# create Logistic regression with base R glm()
myfit <- glm(cardio ~ scale(age), data = mydata, family = binomial)  # we scale d age column as usual
myfit     # compare d value of age column with d value in d 1st numpy=array([[]]) in mymodel$weights object above and see that is is very close

mydata$predicted_glm <- predict(myfit, select(mydata, age), type = 'response')   # creates a new column- predicted_glm in d main dataset with predicted values of age column

mydata$class_glm <- ifelse(mydata$predicted_glm < 0.5, 0, 1)  # creates a new column- class_tf in d main dataset based on stated values from predicted_glm column(ie if any value in predicted_glm column is < 0.5, write 0, else write 1)
head(mydata)  # compare class_glm column with age column

# examine confusion matrix of glm model
confusionMatrix(as.factor(c(mydata$class_glm)), 
                as.factor(mydata$cardio), positive = '1')  # converting class_glm column and cardio column(d response variable) to factor datatype and comparing d values btw both columns using a confusion matrix
# we can see that d accuracy value is still low and there is too many misclassifications in d confusion matrix as such dis model can be improved


# export model to hdf5 format to working directory
save_model_hdf5(mymodel, 'model.h5')


#-------------------------------------------------------------------------------------------


# If u close d Rstudio and restart it or you restart dis R Session and u wanto load up d model created earlier, run d codes below 
# load exported model
library(keras)
yrdata <- load_model_hdf5('model.h5')

# print model weights and predict on a dataset
yrdata$weights


# import data file
library(tidyverse)

mydata <- read_csv("cardio_CVDs.csv")
str(mydata)

# standardize d independent columns with numerical values(ie d continuous variables)
mydata$age <- scale(mydata$age)  # age column is d independent variable we are using so we standardize it becuz it has numerical values

# select predictor (x) and dependent variable (y) and convert to matrix becuz Tensorflow only works with vectors, matrix or arrays
x_train <- as.matrix(select(mydata, age))   # using only age column as independent variable 
y_train <- as.matrix(select(mydata, cardio))   # cardio column is response variable


mydata$predicted_tf <- predict(yrdata, x_train)
mydata
