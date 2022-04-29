
library(EBImage)

# make d folder containing d image as yr working directory. If yr image is .png format. just change it to .jpg format
image1 <- readImage("pic1.jpg")  
image2 <- readImage("pic2.jpg")
print(image1)   
                # multiplying dim values 1920*1280*3 = 7372800(which is total number of values in d image)
print(image2)

display(image1)  # displays d image as an interactive image
display(image2)  # displays d image as an interactive image

# plot Intensity Histogram chart
hist(image1)


# Manipulating brightness
a <- image1 + 0.4  # we add 0.2/0.4/0.8 to each value in d array of d image. this will increase d brightness of d image
print(a)
display(a)
hist(a)


b <- image1 - 0.4   # we minus 0.2/0.4/0.8 from each value in d array of d image. this will decrease d brightness of d image
print(b)
display(b)
hist(b)


# Combining 2 images
myc <- combine(image1, image2)  # displays error
display(myc)


# Put 2 images into 1 image
d <- image1 + image2          # displays error
display(d)

          # OR

d <- image1/2 + image2     # we divide d 1st image by 2 or 4 so that d 2nd image is more visible. displays error
display(d)       
hist(d)


# Manipulating Contrast
e <- image1 * 0.4  # we multiply 0.2/0.4/0.8 to each value in d array of d image. this will increase d contrast of d image
print(e)
display(e)
hist(e)


f <- image1 * 3   # we multiply 3/4/5 from each value in d array of d image. this will decrease d contrast of d image
print(f)
display(f)
hist(f)


# Gamma Correction
g <- image1 ^ 0.5   # we multiply each value in d array of d image by power 0.5. this will increase d contrast of d image
print(g)
display(g)
hist(g)


h <- image1 ^ 3   # we multiply each value in d array of d image by power 3. this will reduce d contrast of d image
print(h)
display(h)
hist(h)


# changing d color-mode
colorMode(image1) <- Grayscale    # converts to black and white color-mode
print(image1)            # d colorMode is now Grayscale
display(image1)

colorMode(image1) <- Color   # returns d image back to color mode
display(image1)


# Cropping
print(image1)        # d image dimensions are: x= 1920 y= 1280 z= 3  
k <- image1[200:700, 300:650,]    
display(k)


# Save d cropped image to a new image file
writeImage(k, "NewImage.jpg")  # saves d new image to d working directory


# Flip, Flop, Rotate and Resize d image
l <- flip(image1)  # flips d image upside-down
display(l)

m <- rotate(image1, 45)    # rotates d image 45 degrees
display(m)

o <- flop(image1)   # returns d image to it default orientation
display(o)

n <- resize(image1, 400)    # changes/reduce d scale of d pix to 400px
display(n)


# creating/using Low-pass filter 
low <- makeBrush(81, shape = "disc", step = FALSE)^ 2   # displays 81X81 matrix. u can use any value
low
low <- low / sum(low)

lowpic <- filter2(image1, low)
display(lowpic)         # dis makes d image blurry


# creating/using High-pass filter
high <- matrix(1, nc = 3, nr = 3)
high[2,2] <- -8

highpic <- filter2(image1, high)
display(highpic)                 # creates white outlines on black background

# reducing d High-pass filter
high <- matrix(1, nc = 3, nr = 3)
high[2,2] <- -4

highpic <- filter2(image1, high)  # increases d contrast of d image
display(highpic)

display(highpic+image1)  # has very high contrast. we combine d high-pass filter image with d original image
display(highpic/8+image1)   # contrast is much better


# Combining two images so that u can see d images by clicking Next Frame icon in LHS of Viewer tab 
new <- highpic/8+image1
comb <- combine(image1, new)
display(comb)

