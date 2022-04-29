# Generative Adversarial Networks (G.A.N)
# GANs algorithm is composed of 2 parts- the Generator and the Discriminator

# We feed d system with images dataset, the Generator makes/creates new images based on d dataset, and d Discriminator tries to spot d difference bwt d actual image and d generated image.
# the aim is to fool d Discriminator into thinking that d generated images are actual images

# GANs algorithm can be used for creating new images/artworks, improving quality, creating new text, new music and in anomaly detection.


# begin <- Sys.time()

library(keras)
library(EBImage)

# MNIST data
mydata <- dataset_mnist()    # dis MNIST data is in keras package
str(mydata)   

c(c(trainx, trainy), c(testx, testy)) %<-% mydata   
summary(trainx)
str(trainx)     # we have 60000 images with height- 28 and width- 28
table(trainy)  # we see d total number of images for each digit. we are using digit 5(which has 5421 images) for dis project

trainx <- trainx[trainy == 5,,]  # we are using digit 5(which has 5421 images) for dis project.  ,, means d dimension which is 28 by 28

par(mfrow = c(8,8), mar = rep(0, 4))

for (i in 1:64) plot(as.raster(trainx[i,,], max = 255))  
                                                        # dis are d real images
par(mfrow = c(1,1))

trainx <- array_reshape(trainx, c(nrow(trainx), 28, 28, 1))
str(trainx)   # displays total number of images and dimensions

trainx <- trainx / 255   # dis will make d values range btw 0 and 1
summary(trainx)


# Generator network. Here we create fake images of digit 5
h <- 28; w <- 28; c <- 1; l <- 28   

gi <- layer_input(shape = l)

go <- gi %>% layer_dense(units = 32 * 14 * 14) %>%
  layer_activation_leaky_relu() %>% 
  layer_reshape(target_shape = c(14, 14, 32)) %>% 
  layer_conv_2d(filters = 32, kernel_size = 5,
                padding = "same") %>%    # dis is a convolutional layer  
  layer_activation_leaky_relu() %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = 4,
                          strides = 2, padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 1, kernel_size = 5,
                activation = "tanh", padding = "same")

g <- keras_model(gi, go)    # this will create fake images of digit 5 and all will have 28 by 28 by 1 dimensions which are values for conv2d (Conv2D) parameter under Layer(type column)
summary(g)


# Discriminator Network
di <- layer_input(shape = c(h, w, c))

do <- di %>% 
  layer_conv_2d(filters = 64, kernel_size = 4) %>% 
  layer_activation_leaky_relu() %>% 
  layer_flatten() %>%
  layer_dropout(rate = 0.3) %>%  
  layer_dense(units = 1, activation = "sigmoid")   # dis is d classification layer dat classifies d images as fake or real

d <- keras_model(di, do)
summary(d)   # d dimension values in input_2 (InputLayer) parameter under Layer(type column) must match dimension values in conv2d (Conv2D) parameter in summary(g) made earlier


# Compile d Discriminator Network
d %>% compile(optimizer = 'rmsprop',
              loss = "binary_crossentropy")


# Freeze weights and Compile
freeze_weights(d) 

gani <- layer_input(shape = l)

gano <- gani %>% g %>% d

gan <- keras_model(gani, gano)

gan %>% compile(optimizer = 'rmsprop', 
                loss = "binary_crossentropy")


# set your working directory before running d code below: goto Session menu > Set Working directory > choose directory#

dir <- "images"   # name of folder to save dis fake images in in yr working directory

dir.create(dir)  # dis will create a folder called images in yr working directory


# Train d G.A.N network
start <- 1; dloss <- NULL; gloss <- NULL   # dloss means discriminated loss, gloss means generated loss


# Generate 50 fake images -----
b <- 50    # this is d batch_size


for (i in 1:100) {
  noise <- matrix(rnorm(b * l), nrow = b, ncol= l)  # rnorm is random normal distribution 

# you can add a closing curly-braces bracket for dis for-loop here and keep on running downwards till u get to # Saves d fake images, then come back here and remove d ending curly-braces bracket and add it in # Saves d fake images  
# then re-run from # Train d G.A.N network above downwards
#------------------------------
  
fake <- g %>% predict(noise)
str(noise)
str(fake)     # d fake images is stored here


# Combine real and fake images
# start <- 1
stop <- start + b - 1  # where start is 1
stop

real <- trainx[start:stop,,,]  # ,,, is for d 3 dimensions in str(fake)

real <- array_reshape(real, c(nrow(real), 28, 28, 1))

rows <- nrow(real)
rows         # displays 50 

both <- array(0, dim = c(rows * 2, dim(real)[-1]))  

both[1:rows,,,] <- fake  # 1st 50 images in rows will be fake images

both[(rows + 1):(rows * 2),,,] <- real  # from 51st to 100th images will be for real images

labels <- rbind(matrix(runif(b, 0.9,1), nrow = b, ncol = 1),   # runif means random numbers with uniform distribution. value 1 denotes fake image but we wont write 1 rather we will be randomize it/use a range btw 0.9 and 1. 
                matrix(runif(b, 0, 0.1), nrow = b, ncol = 1))  # value 0 denotes real image but we will not write 0 but we will randomize it/use a range btw 0 and 0.1

labels

start <- start + b


# Train d Discriminator Network
dloss[i] <- d %>% train_on_batch(both, labels)   # we will train d images in batches. both object contain d 50 real and 50 fake images


# Train Generator using G.A.N
fakeAsReal <- array(runif(b, 0, 0.1), dim = c(b, 1))  # we are trying to fool d network by using a range btw 0 and 0.1(which is d range for real images) on d fake images
str(fakeAsReal)

gloss[i] <- gan %>% train_on_batch(noise, fakeAsReal) 


# Saves d fake images
f <- fake[1,,,]   # saves d 1st fake images from each iteration. It will make 100 iteration based on d for-loop in # Generate 50 fake images way above 
dim(f) <- c(28,28,1)  # dimension of d fake image
image_array_save(f, path = file.path(dir,           # saves d 1st fake images from each iteration ind folder we created earlier
                                     paste0("f", i, ".png")))}   
#-----------------------------



# Plot dloss
x <- 1:100

plot(x, dloss, col = 'red', type = 'l',
     ylim = c(0, 3), xlab = 'Iterations',
     ylab = 'Loss')

lines(x, gloss, col = 'black', type = 'l')

legend('topright', 
       legend = c("Discriminator Loss", "GAN Loss"),
       col = c("red", 'black'), lty = 1:2, cex = 1)


# lets view d 100 fake images in stored in d folder in our working directory
# set your working directory to d folder created earlier that contains d fake images #
temp = list.files(pattern = "*.png")  # dis will get all d images in d folder
temp

mypic <- list()

for (i in 1:length(temp))  {
  mypic[[i]] <- readImage(temp[[i]])
  }

par(mfrow = c(10,10))

for (i in 1:length(temp)) plot(mypic[[i]])   # displays d fake images from d folder in yr working directory 





#---------------------------------------------------------------------------------------------------------------




# lets improve the Generator network by adding anoda convolutional layer to the Generator network and Discriminator network

# Generator network. Here we create fake images of digit 5
h <- 28; w <- 28; c <- 1; l <- 28   

gi <- layer_input(shape = l)

go <- gi %>% layer_dense(units = 32 * 14 * 14) %>%
  layer_activation_leaky_relu() %>% 
  layer_reshape(target_shape = c(14, 14, 32)) %>% 
  layer_conv_2d(filters = 32, kernel_size = 5,
                padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d_transpose(filters = 32, kernel_size = 4,
                          strides = 2, padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 64, kernel_size = 5,  
                padding = "same") %>%      # just added dis
  layer_activation_leaky_relu() %>%       # and dis
  layer_conv_2d(filters = 1, kernel_size = 5,
                activation = "tanh", padding = "same")

g <- keras_model(gi, go)    
summary(g)



# Discriminator Network. adding anoda convolutional layer to the Generator network
di <- layer_input(shape = c(h, w, c))

do <- di %>% 
  layer_conv_2d(filters = 64, kernel_size = 4) %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 64, kernel_size = 4,
                strides = 2) %>%     # just added dis
  layer_activation_leaky_relu() %>%   # and dis
  layer_flatten() %>%
  layer_dropout(rate = 0.3) %>%  
  layer_dense(units = 1, activation = "sigmoid")   # dis is d classification layer dat classifies d images as fake or real

d <- keras_model(di, do)
summary(d)   


# Compile d Discriminator Network
d %>% compile(optimizer = 'rmsprop',
              loss = "binary_crossentropy")


# Freeze weights and Compile
freeze_weights(d) 

gani <- layer_input(shape = l)

gano <- gani %>% g %>% d

gan <- keras_model(gani, gano)

gan %>% compile(optimizer = 'rmsprop', 
                loss = "binary_crossentropy")


# set your working directory before running d code below: goto Session menu > Set Working directory > choose directory#

dir <- "images2"   # name of folder to save dis fake images in in yr working directory

dir.create(dir)  # dis will create a folder called images in yr working directory


# Train d G.A.N network
start <- 1; dloss <- NULL; gloss <- NULL   # dloss means discriminated loss, gloss means generated loss


# Generate 50 fake images -----
b <- 50    # this is d batch_size


for (i in 1:100) {
  noise <- matrix(rnorm(b * l), nrow = b, ncol= l)  # rnorm is random normal distribution 
  

fake <- g %>% predict(noise)
str(noise)
str(fake)     # d fake images is stored here
  
  
# Combine real and fake images
# start <- 1
stop <- start + b - 1  # where start is 1
stop
  
real <- trainx[start:stop,,,]  # ,,, is for d 3 dimensions in str(fake)

real <- array_reshape(real, c(nrow(real), 28, 28, 1))
  
rows <- nrow(real)
rows         # displays 50 
  
both <- array(0, dim = c(rows * 2, dim(real)[-1]))  
# both object will contain 50 fake and 50 real images
  
both[1:rows,,,] <- fake  # 1st 50 images in rows will be fake images
  
both[(rows + 1):(rows * 2),,,] <- real  # from 51st to 100th images will be for real images
  
labels <- rbind(matrix(runif(b, 0.9,1), nrow = b, ncol = 1),    
                  matrix(runif(b, 0, 0.1), nrow = b, ncol = 1))  # value 0 denotes real image but we will not write 0 but we will randomize it/use a range btw 0 and 0.1
  
labels
  
start <- start + b
  
  
# Train d Discriminator Network
dloss[i] <- d %>% train_on_batch(both, labels)   # we will train d images in batches. both object contain d 50 real and 50 fake images
  
  
# Train Generator using G.A.N
fakeAsReal <- array(runif(b, 0, 0.1), dim = c(b, 1))  # we are trying to fool d network by using a range btw 0 and 0.1(which is d range for real images) on d fake images
str(fakeAsReal)
  
gloss[i] <- gan %>% train_on_batch(noise, fakeAsReal) 
  
  
# Saves d fake images
f <- fake[1,,,]   # saves d 1st fake images from each iteration. It will make 100 iteration based on d for-loop in # Generate 50 fake images way above 
dim(f) <- c(28,28,1)  # dimension of d fake image
image_array_save(f, path = file.path(dir,           # saves d 1st fake images from each iteration in d folder we created earlier
                                     paste0("f", i, ".png")))}   




# Plot dloss
x <- 1:100

plot(x, dloss, col = 'red', type = 'l',
     ylim = c(0, 3), xlab = 'Iterations',
     ylab = 'Loss')

lines(x, gloss, col = 'black', type = 'l')

legend('topright', 
       legend = c("Discriminator Loss", "GAN Loss"),
       col = c("red", 'black'), lty = 1:2, cex = 1)


# lets view d 100 fake images in stored in d folder in our working directory
# set your working directory to d folder created earlier that contains d fake images #
temp = list.files(pattern = "*.png")  # dis will get all d images in d folder
temp

mypic <- list()

for (i in 1:length(temp))  {
  mypic[[i]] <- readImage(temp[[i]])
}

par(mfrow = c(10,10))

for (i in 1:length(temp)) plot(mypic[[i]])   # displays d fake images from d folder in yr working directory 






# start <- start + b
# 
# if (start > (nrow(trainx) - b))  start <- 1
# 
# cat("Discriminator Loss:", dloss[i], "\n") 
# cat("Gan Loss:", gloss[i], "\n")  


# #Time
# end <- Sys.time()
# begin
# end
# end - begin