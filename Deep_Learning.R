# creating a Deep Learning model to make predictions on d response variable in d dataset. All columns in d dataset must be numerical/integer datatype for u to use Deep Learning model 

library(keras)

mydata <- read.csv(file.choose(), header = T)  # select Cardiotocographic.csv
str(mydata)   # all d datatype of d columns are int/numerical so we can continue


# change d dataset to a matrix
mydata <- as.matrix(mydata)
dimnames(mydata) <- NULL     # removes all d column headers names
str(mydata)


# Normalize d independent variable/columns
mydata[,1:21] <- normalize(mydata[,1:21])  # normalize() is a tensorflow function. d dependent column is d 22nd column so it is not included in d Normalization

mydata[,22] <- as.numeric(mydata[,22]) -1  

summary(mydata)  


# Data Partition
set.seed(1234)
ind <- sample(2, nrow(mydata), replace = TRUE, prob = c(0.7,0.3))
training <- mydata[ind == 1, 1:21]   # we move value 1 in all d independent variables into training data
testing <- mydata[ind == 2, 1:21]   # we move value 2 in all d independent variables into testing data

trainingtarget <- mydata[ind == 1, 22]   # we move value 1 in d dependent variable into trainingtarget data
testingtarget <- mydata[ind == 2, 22]   # we move value 2 in d dependent variable into testingtarget data
head(testingtarget, 10)


# One Hot Encoding: dis is converting d dependent column with its 3 distinct values into dummy variables/column with values 0s and 1s
trainlabel <- to_categorical(trainingtarget)   
testlabel <- to_categorical(testingtarget)   
testlabel   


# create sequential model
mymodel <- keras_model_sequential()

mymodel %>%
  layer_dense(units = 8, activation = 'relu', input_shape = c(21)) %>%  
  layer_dense(units = 3, activation = 'softmax')   
                             

summary(mymodel)   


# Compile d model to configure d learning process
mymodel %>%
  compile(loss = 'categorical_crossentropy', optimizer = 'adam',
          metrics = 'accuracy')  


# Fit d model
ourmodel <-mymodel %>%
  fit(training, trainlabel, epoch = 200, batch_size = 32,    # epoch = 200 means we run/iterate d model X200.
      validation_split = 0.2)    

plot(ourmodel)   # top chart is for Loss and bottom chart is for Accuracy


# Evaluate d model using testing data
yrmodel <- mymodel %>% evaluate(testing, testlabel)
yrmodel    # displays Loss and Accuracy values



# Prediction and Confusion Matrix for testing data
# mypred <- mymodel %>% predict_proba(testing)     predict_proba() and predict_classes() are no longer in use in d latest Tensorflow
# mypred1 <- mymodel %>% predict_classes(testing)

mypred <- mymodel %>% predict(testing)  # use predict() in place of predict_proba()   

mypred1 <- mymodel %>%     # use predict() %>% k_argmax() %>% as.numeric() in place of predict_classes(). using k_argmax() becuz we used activation = softmax in d sequential model above. softmax is used for multi-class classification(ie we have 3 distinct values in d response variable)
 predict(testing) %>%    # If we were using activation = sigmoid(which is binary classification) in d sequential model above, we write-  mypred1 <- mymodel %>% predict(testing) %>% `>`(0.5) %>% k_cast("int32")
 k_argmax() %>%
  as.numeric()    # converts to numeric datatype so it is same datatype with testingtarget object that will be used to create confusion matrix

# class(mypred1)   # mypred1 and testingtarget objects are both same datatype
# class(testingtarget)


tab <- table(Predicted = mypred1, Actual = testingtarget)   
tab

# traceback()   # running dis after an error traces back to how d error was made

cbind(mypred, mypred1, testingtarget)   # index 6 is a misclassification since mypred1 and testingtarget does not have d same values


# Fine-tune d model
tab
yrmodel  # displays accuracy and loss values




#---------------------------------------------------------------------------------------------




# improving d model accuracy starting from d sequential model created
# create sequential model
mymodel <- keras_model_sequential()

mymodel %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(21)) %>%  
  layer_dense(units = 3, activation = 'softmax')   

summary(mymodel)   


# Compile d model to configure d learning process
mymodel %>%
  compile(loss = 'categorical_crossentropy', optimizer = 'adam',
          metrics = 'accuracy')  


# Fit d model
ourmodel <-mymodel %>%
  fit(training, trainlabel, epoch = 200, batch_size = 32,    # epoch = 200 means we run/iterate d model X200.
      validation_split = 0.2)    

plot(ourmodel)   # top chart is loss and bottom chart is for accuracy


# Evaluate d model using testing data
yrmodel1 <- mymodel %>% evaluate(testing, testlabel)
yrmodel1


# Prediction and Confusion Matrix for testing data
mypred <- mymodel %>% predict(testing)  # use in place of predict_proba()   

mypred1 <- mymodel %>%     # using dis in place of predict_classes(). using k_argmax() becuz we used activation = softmax in d sequential model above. softmax is used for multi-class classification(ie we have 3 distinct values in d response variable)
  predict(testing) %>%    # If we were using activation = sigmoid(which is binary classification) in d sequential model above, we write-  mypred1 <- mymodel %>% predict(testing) %>% `>`(0.5) %>% k_cast("int32")
  k_argmax() %>%
  as.numeric()    # converts to numeric datatype so it is same datatype with testingtarget object that will be used to create confusion matrix


tab1 <- table(Preducted = mypred1, Actual = testingtarget)
tab1

cbind(mypred, mypred1, testingtarget)


# Fine-tune d model
tab1      # compare with tab object from d 1st model
yrmodel1   # compare with d accuracy and loss values in yrmodel object from d 1st model





#----------------------------------------------------------------------------------------------------------------







# improving d model accuracy starting from d sequential model created
# create sequential model
mymodel <- keras_model_sequential()

mymodel %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(21)) %>%  #
  layer_dense(units = 8, activation = 'relu') %>%    s
  layer_dense(units = 3, activation = 'softmax')   

summary(mymodel)   


# Compile d model to configure d learning process
mymodel %>%
  compile(loss = 'categorical_crossentropy', optimizer = 'adam',
          metrics = 'accuracy')  


# Fit d model
ourmodel <-mymodel %>%
  fit(training, trainlabel, epoch = 200, batch_size = 32,    # epoch = 200 means we run/iterate d model X200.
      validation_split = 0.2)    #

plot(ourmodel)   # top chart is loss and bottom chart is for accuracy


# Evaluate d model using testing data
yrmodel2 <- mymodel %>% evaluate(testing, testlabel)
yrmodel2


# Prediction and Confusion Matrix for testing data
mypred <- mymodel %>% predict(testing)  # use in place of predict_proba()   

mypred1 <- mymodel %>%     # using dis in place of predict_classes(). using k_argmax() becuz we used activation = softmax in d sequential model above. softmax is used for multi-class classification(ie we have 3 distinct values in d response variable)
  predict(testing) %>%    # If we were using activation = sigmoid(which is binary classification) in d sequential model above, we write-  mypred1 <- mymodel %>% predict(testing) %>% `>`(0.5) %>% k_cast("int32")
  k_argmax() %>%
  as.numeric()    # converts to numeric datatype so it is same datatype with testingtarget object that will be used to create confusion matrix

tab2 <- table(Preducted = mypred1, Actual = testingtarget)

cbind(mypred, mypred1, testingtarget)


# Fine-tune d model
tab2      # compare with tab and tab1 objects from 1st and 2nd  model
yrmodel2   # compare with d accuracy and loss values in yrmodel and yrmodel1 object from 1st and 2nd model

#------------------------------------------------------------------------------------------


# comparing to find which has d highest accuracy. we see that tab2/yrmodel2 has d highest accuracy
tab
yrmodel

tab1      # compare with tab object from d 1st model
yrmodel1

tab2      # compare with tab and tab1 objects from 1st and 2nd model
yrmodel2
