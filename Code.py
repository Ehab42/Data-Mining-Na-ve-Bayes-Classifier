import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import os




# Reading training image set in an array named 'image'
path = "./Train"
filenames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))


# For simplicity, dividing each pixel in each image by 255, and save the result in the array 'newImages'
newImages = [[[0 for i in range(0,12)] for i in range(0,12)] for i in range(0,len(images))]
for i in range(0,len(images)):
    for a in range(0,12):
        for b in range(0,12):
            temp = images[i][a][b]/float(255)
            newImages[i][a][b] = temp

# Reading testing image set in an array named 'testImages'
path = "./Test"
filenames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
testImages = []
for file in filenames:
    testImages.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))

# For simplicity, dividing each pixel in each image by 255, and save the result in the array 'newTestImages'
newTestImages = [[[0 for i in range(0,12)] for i in range(0,12)] for i in range(0,len(testImages))]
for i in range(0,len(testImages)):
    for a in range(0,12):
        for b in range(0,12):
            temp = testImages[i][a][b]/float(255)
            newTestImages[i][a][b] = temp

# Saving the name of each imageClass in an array called 'classNames'
classNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# Calculate the mean and variance of each pixel of each image in its corresponding class
# Each mean and variance of each class is a 12*12 array, in which each index corresponds to each pixels' mean and variance.
# Then storing the means and variances of each class in a array called 'means' and 'Variances'
means = [[[0 for i in range(0, 12)] for i in range(0, 12)] for i in range(0, 26)]
Variances = [[[0 for i in range(0, 12)] for i in range(0, 12)] for i in range(0, 26)]
imageClass = []
temp = []
count = 0
for i in range(0, len(newImages), 7):
    for j in range (i, i+7):
        imageClass.append(newImages[j])
    for a in range(0,12):
        for b in range(0,12):

                for image in imageClass:
                    temp.append(image[a][b])

                means[count][a][b] = np.mean(temp)
                Variances[count][a][b] = np.var(temp)
                temp = []
    count = count + 1
    imageClass = []


# Calculating Gaussian Probability
def calcProbGauss(pixelMean, pixelVariance, pixel):
    if pixelVariance < 0.0001:
        pixelVariance = 0.0001
    exponent = math.exp(-(math.pow(pixel-pixelMean, 2)/(2*pixelVariance)))
    likelihood = (1/(math.sqrt(2*math.pi)*pixelVariance))*exponent
    if likelihood < 0.1:
        likelihood = 0.01
    return likelihood

# Calculating the likelehood (P(img|Cnum)) probability of a specific image in a specific class
def calcLiklehoodProbs(img, classNum):
    imgLikeLihood = 1
    for a in range(0,12):
        for b in range(0,12):
            pixelMean = means[classNum][a][b]
            pixelVariance = Variances[classNum][a][b]
            pixel = img[a][b]
            imgLikeLihood *= calcProbGauss(pixelMean, pixelVariance, pixel)
    return imgLikeLihood

# Storing all likelihoods of an image in all classes in the array 'allLikelihoods'
def calcAllLiklehoodProbs(img):
    allLiklehoods = []
    for classNum in range(0, 26):
        allLiklehoods.append(calcLiklehoodProbs(img, classNum))
    return allLiklehoods

# Calculating the probability of each image (P(img))
def probEachImg(img):
    sumProbs = 0
    for j in range(0,26):
        sumProbs = sumProbs + (calcLiklehoodProbs(img, j) * (1/26))
    return sumProbs

# Calculating the posterior probability (P(Cnum|img)) of a specific image in a specific class
def calcPostProb(img, classNum):
    postProb = (calcLiklehoodProbs(img, classNum) * (1/26)) / probEachImg(img)
    return postProb

# This is the final testing method, where it takes an image as a parameter and return its predicted class
def predictImageClass(img):
    allPostProbs = []
    for i in range(0,26):
        allPostProbs.append(calcPostProb(img, i))
    index = allPostProbs.index(max(allPostProbs))
    className = classNames[index]
    return className


# The array 'newTestImages' contain 52 (2*26) images to be tested, ordered each pair in a class (A --> Z)
# In order to test the algorithm enter an index of an image and the predicted class will be printed in the console
index = 7
# For example index: 7, means in the 4th pair which is in class: D
print(predictImageClass(newTestImages[index]))


# A method that returns an array of size 26 called 'accuracy',
# which contains the number of images classified correctly for each class character
def getClassAccuracy():
    count = 0
    output = []
    accuracyaArray = []
    for i in range(0, len(newTestImages)):
        output.append(predictImageClass(newTestImages[i]))

    for j in range(0, 26):
        for i in range(0, 2):
            if output[i] == classNames[j]:
                count = count + 1
        output.pop(0)
        output.pop(0)
        accuracyaArray.append(count)
        count = 0
    return accuracyaArray



# Plotting the accuracy as a histogram
plt.figure(1, figsize=(50, 7))
plt.bar(classNames, getClassAccuracy())
plt.suptitle('Accuracy')
plt.xlabel('Image Classes')
plt.ylabel('Number of images classified correctly for each Class')
plt.show()








