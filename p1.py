#/usr/bin/python3
import os
import scipy as sp
from scipy import misc
import numpy as np 
from matplotlib import pylab, pyplot,colors
import cv2 

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

def showHistogram(inputImage,outputImage):
        pyplot.subplot(2,2,1)
        pyplot.imshow(inputImage,cmap='gray',norm=colors.NoNorm())
        pyplot.title('Original image')
       
        pyplot.subplot(2,2,2)
        pyplot.hist(inputImage.ravel(),256,[0,255])
        pyplot.title('Original hist')
       
        pyplot.subplot(2,2,3)
        pyplot.imshow(outputImage,cmap='gray',norm=colors.NoNorm())
        pyplot.title('Modified')
        
        pyplot.subplot(2,2,4)
        pyplot.hist(outputImage.ravel(),256,[0,255])
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
        
        #outImage=normalizeData(outImage,inRange[1],inRange[0])
        return outImage.astype(int)


        
def equalizeIntensity(inImage, nbins = 256):
        inImage = cv2.imread(PATH+inImage,0)
        arrImage = np.asarray(inImage)
        dimX = len(arrImage)
        dimY = len(arrImage[0]) 
        outImage = np.zeros((dimX,dimY),"f")
        
        histAcum,bins,x =pyplot.hist(arrImage.ravel(),bins=256,cumulative=True)
        hist,bins,x =pyplot.hist(arrImage.ravel(),bins=256)
        cdf = 255 * histAcum/(dimX*dimX)

        for i in range(dimX):
                for j in range(dimY):
                        outImage [i,j]= cdf[arrImage[i,j]] 
        
        
        return outImage.astype(int)
        
        
        
        

        
        
        
   

if __name__ == '__main__':
        
        inImage = cv2.imread('pictures/lena.png',0)
        #outImage = adjustIntensity('lena.png')
        outImage = equalizeIntensity('lena.png')
        #showImage(inImage,outImage)
        showHistogram(inImage,outImage)
        print(outImage)
        cv2.imwrite("pictures/outImage.png",outImage)
        #norm_image = cv2.normalize(outImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #cv2.imwrite("pictures/outImage.png",norm_image)
        
        
