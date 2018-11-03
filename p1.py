#/usr/bin/python3
import os
import scipy as sp
from scipy import misc
import numpy as np 
from matplotlib import pylab, pyplot,colors
import cv2 
import math

#path de la carpeta donde guardo las imagenes
PATH = os.path.abspath('pictures') + '/'

#Funciones que uso para la visualizar una imagen con pyplot
def showImage(inputImage,outputImage):
        pyplot.subplot(1,2,1)
        pyplot.imshow(inputImage,cmap='gray',norm=colors.NoNorm())
        pyplot.title('Original')
        pyplot.subplot(1,2,2)
        pyplot.imshow(outputImage,cmap='gray',norm=colors.NoNorm())
        pyplot.title('Modified')
        pyplot.show()

def showHistogram(inputImage,outputImage,bins=[256,256]):
        pyplot.subplot(2,2,1)
        pyplot.imshow(inputImage,cmap='gray',norm=colors.NoNorm())
        pyplot.title('Original image')
       
        pyplot.subplot(2,2,2)
        pyplot.hist(inputImage.ravel(),bins[0],[0,bins[0]])
        pyplot.title('Original hist')
       
        pyplot.subplot(2,2,3)
        pyplot.imshow(outputImage,cmap='gray',norm=colors.NoNorm())
        pyplot.title('Modified')
        
        pyplot.subplot(2,2,4)
        pyplot.hist(outputImage.ravel(),bins[0],[0,bins[1]])
        pyplot.title('Modified')
        pyplot.show()
        

def normalizeData(inputData,maxValue,minValue):
        dimX = len(inputData)
        dimY = len(inputData[0]) 
        normData = np.zeros((dimX,dimY),"f")
        #maxValue = np.amax(inputData)
        #minValue = np.amin(inputData)
        print(maxValue,minValue)
        for i in range(dimX):
                for j in range(dimY):
                        normData[i,j] = (inputData[i,j]-minValue) / (maxValue-minValue)

        return normData 
        
#Modificar rango dínamico
def adjustIntensity(inImage,inRange =[], outRange = [0,1]):
        inImage = cv2.imread((PATH+inImage),0)
        arrImage = np.asarray(inImage)
        maxValue = np.amax(arrImage)
        minValue = np.amin(arrImage)

        
        if (inRange == []):
                inRange = np.array([minValue,maxValue],dtype=int)
                
        
        dimX = len(arrImage)
        dimY = len(arrImage[0]) 
        outImage = np.zeros((dimX,dimY),"f")
        
        for i in range(dimX):
                for j in range(dimY):
                        outImage[i,j] = inRange[0] + ((inRange[1]-inRange[0])*((arrImage[i,j]-minValue)) / (maxValue-minValue)) 
        
        
        return outImage.astype(int)


        
def equalizeIntensity(inImage, nbins = 256):
        inImage = cv2.imread(PATH+inImage,0)
        arrImage = np.asarray(inImage)
        dimX = len(arrImage)
        dimY = len(arrImage[0]) 
        outImage = np.zeros((dimX,dimY),"f")
        
        histAcum,bins,x =pyplot.hist(arrImage.ravel(),bins=256,cumulative=True)
        hist,bins,x =pyplot.hist(arrImage.ravel(),bins=256)
        """
        histAcum = np.zeros(1,"f")
        sum = 0
        for value in hist:
                sum += value
                histAcum.append(sum)
        """
        cdf = (histAcum/(dimX*dimX)) *255
        for i in range(dimX):
                for j in range(dimY):
                        outImage [i,j]= cdf[arrImage[i,j]] 
        
        
        return outImage.astype(int)
        
        
        
def filterImage(inImage, kernel):
        kernelP = len(kernel)
        kernelQ = len(kernel[0])
        centerK = (kernelP//2),(kernelQ//2)
        c1 = kernelP//2
        c2 = kernelQ//2
        #print(centerK)
        arrImage=inImage#.asarray(inImage)
        dimX = len(arrImage)
        dimY = len(arrImage[0])
        #print(dimX,dimY)
        outImage = np.zeros((dimX,dimY),float)
        convImage = np.zeros((dimX+2,dimY+2),float)
        convImage[1:-1,1:-1] = inImage
        for i in range(dimX):
                for j in range(dimY):
                        outImage[i,j] = (kernel * convImage[i:i+kernelP,j:j+kernelQ]).sum()
                                     
                                      
                
                
        return outImage
        
   

if __name__ == '__main__':
        
       # inImage = cv2.imread('pictures/lena.png',0)
        inImage = np.array([[45,60,98,127,132,133,137,133], \
                  [46,65,98,123,126,128,131,133], \
                  [47,65,96,115,119,123,135,137], \
                  [47,63,91,107,113,122,138,134],\
                  [50,59,80,97,110,123,133,134], \
                  [49,53,68,83,97,113,128,133],\
                  [50,50,58,70,84,102,116,126],\
                  [50,50,52,58,69,86,101,120]])
        kernel = np.array([[0.1,0.1,0.1],\
                        [0.1,0.2,0.1],\
                        [0.1,0.1,0.1]])

        
        
        outImage = filterImage(inImage,kernel)
        print(np.round(outImage[1:7,1:7]))
        
        