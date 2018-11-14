#/usr/bin/python3
#Autor: Iago Fernandez Picos
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
        pyplot.hist(outputImage.ravel(),255,[0,bins[1]])
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
       # inImage = cv2.imread((PATH+inImage),0)
        arrImage = normalizeData(inImage,255,0)
        maxValue = np.amax(arrImage)
        minValue = np.amin(arrImage)
        print('max: ',maxValue,'min ',minValue)
        
        maxValue = np.amax(arrImage)
        minValue = np.amin(arrImage)
        if (inRange == []):
                inRange = np.array([minValue,maxValue],dtype=float)
                
        dimX = len(arrImage)
        dimY = len(arrImage[0]) 
        outImage = np.zeros((dimX,dimY),"f")
        
        for i in range(dimX):
                for j in range(dimY):
                     outImage[i,j] = outRange[0] + ((outRange[1]-outRange[0])*((arrImage[i,j]- inRange[0])) / (inRange[1]-inRange[0]))
        
        """
        for i in range(dimX):
                for j in range(dimY):
                        outImage1[i,j] = outRange[0] + ((outRange[1]-outRange[0])*((outImage[i,j]-minValue)) / (maxValue-minValue))
        """
        
        print(outImage)
        return outImage



        
def equalizeIntensity(inImage, nbins = 256):
        #inImage = cv2.imread(PATH+inImage,0)
        arrImage = inImage
        #arrImage = np.round(arrImage,decimals=0)
        #print(arrImage)
        dimX = len(arrImage)
        dimY = len(arrImage[0]) 
        outImage = np.zeros((dimX,dimY),"f")
        
        histAcum,bins,x =pyplot.hist(arrImage.ravel(),nbins,cumulative=True)
        hist,bins,x =pyplot.hist(arrImage.ravel(),nbins)

        cdf = (histAcum/(dimX*dimX))*255
        
        for i in range(dimX):
                for j in range(dimY):
                        outImage [i,j]= cdf[arrImage[i,j]] 
        
        return outImage
        
        
def filterImage(inImage, kernel):
        dim = kernel.shape
        kernelP = dim[0]
        kernelQ = dim[1]
        c1 = (kernelP//2) 
        c2 = (kernelQ//2) 
        dimX = len(inImage)
        dimY = len(inImage[0])
        #Filas y columnas que se añaden a mayores
        leftX = c1
        rightX = kernelP-c1-1
        upY = c2
        downY = kernelQ-c2-1
        convMatrix = np.lib.pad(inImage,((leftX,rightX),(upY,downY)),'constant',constant_values = 0)
        
        outImage = np.zeros((dimX,dimY),"f")
        
        for i in range(dimX):
                for j in range(dimY):
                        outImage[i,j] = (kernel * convMatrix[i:i+kernelP,j:j+kernelQ]).sum()
        
        return outImage.astype(int)
        

        #return outImage
def gaussKernel1D(sigma):
        N = 2*(3*sigma)+1
        N = int((round(N,0)))
        c1 = N//2
        kernel = np.zeros((1,N),"f")
        aux = N%2
        for i in range(N):
                pos = (N-aux)-c1-i
                kernel[0,i] = ((math.pow(math.e,(-(pos**2)/(2*sigma**2)))) / (sigma*math.sqrt((2*math.pi))))
                     
        return kernel

def gaussianFilter(inImage,sigma):
        image = np.asarray(inImage)
        kernel = gaussKernel1D(sigma)
        kernelT = np.transpose(kernel)
        outImage = filterImage(image,kernel)
        outImage = filterImage(outImage,kernelT)
        return outImage.astype(int)
        
 
def medianFilter(inImage, filterSize):
        kernelP = filterSize
        kernelQ = filterSize
        c1 = (kernelP//2) 
        c2 = (kernelQ//2) 
        dimX = len(inImage)
        dimY = len(inImage[0])
        #Filas y columnas que se añaden a mayores
        leftX = c1
        rightX = kernelP-c1-1
        upY = c2
        downY = kernelQ-c2-1
        convMatrix = np.lib.pad(inImage,((leftX,rightX),(upY,downY)),'constant',constant_values = np.nan)
        
        outImage = np.zeros((dimX,dimY),"f")
        
        for i in range(dimX):
                for j in range(dimY):
                        outImage[i,j] = np.nanmedian((convMatrix[i:i+kernelP,j:j+kernelQ]))
        
        return np.round(outImage).astype(int)


def highBoost(inImage,A,method,param):
        
        
        if (method == 'gaussian'):
                print('gaussian')
                smoothImage = gaussianFilter(inImage,param)
        elif (method == 'median'):
                print('median')
                smoothImage = medianFilter(inImage,param)
        
        #smoothImage = adjustIntensity(smoothImage,[0,255])
        #outImage = adjustIntensity(outImage,[0,255])
        
        fImage = A * inImage
        #fImage = adjustIntensity(fImage,[0,255])
       # smoothImage = adjustIntensity(smoothImage,[0,255])
        outImage = fImage-smoothImage
        
        outimage= adjustIntensity(outImage,[0,255])
        #outImage = inImage+outImage
        return np.abs(outImage.astype(int))

def _toBinary(inImage):
        dimX = len(inImage)
        dimY = len(inImage[0])
        outImage = np.zeros((dimX,dimY),int)
        for i in range(dimX):
                for j in range(dimY):
                        if(inImage[i,j] < 20):
                                outImage[i,j] = 0
                        elif(inImage[i,j]>=20):
                                outImage[i,j] = 1
        return outImage

def _toNormal(inImage):
        dimX = len(inImage)
        dimY = len(inImage[0])
        outImage = np.zeros((dimX,dimY),int)
        for i in range(dimX):
                for j in range(dimY):
                        if(inImage[i,j]== 1):
                                outImage[i,j] = 255
        return outImage
def _erode(inImage,kernel):
        dimX = len(inImage)
        dimY = len(inImage[0])
        resultado = 1
        for i in range(dimX):
                for j in range(dimY):
                     if(kernel[i,j] == 1 and inImage[i,j] == 0):
                             resultado = 0
                             break
                if(not resultado):
                        break
        return resultado
def erode(inImage,SE,center = []):
        P = SE.shape[0]
        Q = SE.shape[1]
        
        if (center == []):
                center = [P//2,Q//2]
        c1 = center[0] 
        c2 = center[1]

        dimX = len(inImage)
        dimY = len(inImage[0])
        #Filas y columnas que se añaden a mayores
        leftX = c1
        rightX = P-c1-1
        upY = c2
        downY = Q-c2-1
        binrayImage = _toBinary(inImage)
        if (center == []):
                center = [c1,c2]
         
        convMatrix = np.lib.pad(binrayImage,((leftX,rightX),(upY,downY)),'edge')#,constant_values=0)
        outImage = np.zeros((dimX,dimY),int)
        #SE = np.flip(SE)
        
        for i in range(dimX):
                for j in range(dimY):
                       if ((SE <= convMatrix[i:i+P,j:j+Q]).all()):
                                outImage[i,j] = 1
                       else:
                                outImage[i,j] = 0
                       """
                       if((SE & convMatrix[i:i+P,j:j+Q]).all()):
                                outImage[i,j] = 1
                       else:
                                outImage[i,j]= 0
                      """
                            
        
        return _toNormal(outImage.astype(int))
        

def erode2(inImage,SE,center = []):
        P = SE.shape[0]
        Q = SE.shape[1]
        
        if (center == []):
                center = [P//2,Q//2]
        c1 = center[0] 
        c2 = center[1]

        dimX = len(inImage)
        dimY = len(inImage[0])
        #Filas y columnas que se añaden a mayores
        leftX = c1
        rightX = P-c1-1
        upY = c2
        downY = Q-c2-1
        binrayImage = inImage
        if (center == []):
                center = [c1,c2]
         
        convMatrix = np.lib.pad(binrayImage,((leftX,rightX),(upY,downY)),'edge')
        outImage = np.zeros((dimX,dimY),int)
        #SE = np.flip(SE)
        
        for i in range(dimX):
                for j in range(dimY):
                       outImage[i,j]= np.min(SE*convMatrix[i:i+P,j:j+Q])
                            
        
        return outImage.astype(int)


def _dilatation(inImage,kernel):
        dimX = len(inImage)
        dimY = len(inImage[0])
        resultado = 0
        for i in range(dimX):
                for j in range(dimY):
                     if((kernel[i,j] == 1 and inImage[i,j]) == 1):
                             resultado = 1
                             break
                if(not resultado):
                        break
        return resultado

def dilatation(inImage,SE,center = []):
        P = SE.shape[0]
        Q = SE.shape[1]
        c1 = (P//2) 
        c2 = (Q//2) 
        dimX = len(inImage)
        dimY = len(inImage[0])
        #Filas y columnas que se añaden a mayores
        leftX = c1
        rightX = P-c1-1
        upY = c2
        downY = Q-c2-1
        binrayImage = _toBinary(inImage)
        if (center == []):
                center = [c1,c2]
         
        convMatrix = np.lib.pad(binrayImage,((leftX,rightX),(upY,downY)),'constant',constant_values=0)
        
        outImage = np.zeros((dimX,dimY),"f")
        SE = np.flip(SE)
        for i in range(dimX):
                for j in range(dimY):
                        for q in range(i,i+P):
                                outImage[i,j] = _dilatation(convMatrix[i:i+P,j:j+Q],SE)
                                
                                """"
                                if ((SE & convMatrix[i:i+P,j:j+Q]).any() ):
                                        outImage[i,j] = 1
                                else:
                                        outImage[i,j] = 0
                                """
        return _toNormal(outImage)
        
def opening(inImage,SE,center):
        midImage = erode(inImage,SE,center)
        outImage = dilatation(midImage,SE,center)

        return outImage

def closing(inImage,SE,center):
        midImage = dilatation(inImage,SE,center)
        outImage = erode(midImage,SE,center)

        return outImage

def complementario(inImage):
        dimX = len(inImage)
        dimY = len(inImage[0])
        outImage = np.zeros((dimX,dimY),int)
        for i in range(dimX):
                for j in range(dimY):
                        if inImage[i,j] == 0:
                                outImage[i,j] = 1
                        else:
                                outImage[i,j] = 0
        return inImage
def hit_or_miss(inImage,objSEj,bgSE,center):
        dimX = len(inImage)
        dimY = len(inImage[0])

        P = len(objSEj)
        Q = len(objSEj[0])
        
        print(inImage)
        comIm = complementario(inImage)

        for p in range(P):
                for q in range(Q):
                        if (objSEj[p,q] == 1 and bgSE[p,q] == 1):
                                print('Error: Elementos estructurantes incoherentes')
                                exit
        outImage1 = erode(_toNormal(inImage),objSEj,center)
        outImage2 = erode(_toNormal(comIm),bgSE,center)
        outImage = outImage1 & outImage2
        return _toNormal(outImage)

def _Roberts(inImage):
        dimX = len(inImage)
        dimY = len(inImage[0])
        kernelX = np.array([[-1,0],[0,1]])
        kernelY = np.array([[0,-1],[1,0]])
        P = len(kernelX)
        Q = len(kernelX[0])
        Gx = np.zeros((dimX,dimY),int)
        Gy = np.zeros((dimX,dimY),int)
        for i in range(dimX-P):
                for j in range(dimY-Q):
                        Gx[i,j] = (kernelX * inImage[i:i+P,j:j+Q]).sum()
                        Gy[i,j] = (kernelY * inImage[i:i+P,j:j+P]).sum()
        return np.abs(Gx),np.abs(Gy)

def _CentralDiff(inImage):
        dimX = len(inImage)
        dimY = len(inImage[0])
        kernelX = np.array([[-1,1],[0,0]])
        kernelY = np.array([[-1,0],[1,0]])
        P = len(kernelX)
        Q = len(kernelX[0])
        Gx = np.zeros((dimX,dimY),int)
        Gy = np.zeros((dimX,dimY),int)
        for i in range(dimX-P):
                for j in range(dimY-Q):
                        Gx[i,j] = (kernelX * inImage[i:i+P,j:j+Q]).sum()
                        Gy[i,j] = (kernelY * inImage[i:i+P,j:j+P]).sum()
        return np.abs(Gx),np.abs(Gy)


def _Prewitt(inImage):
        dimX = len(inImage)
        dimY = len(inImage[0])
        kernelX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        kernelY = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        P = len(kernelX)
        Q = len(kernelX[0])
        Gx = np.zeros((dimX,dimY),int)
        Gy = np.zeros((dimX,dimY),int)
        
        for i in range(dimX-P):
                for j in range(dimY-Q):
                        Gx[i,j] = (kernelX * inImage[i:i+P,j:j+Q]).sum()
                        Gy[i,j] = (kernelY * inImage[i:i+P,j:j+Q]).sum()
        return np.abs(Gx),np.abs(Gy)

def _Sobel(inImage):
        dimX = len(inImage)
        dimY = len(inImage[0])
        kernelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        kernelY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        P = len(kernelX)
        Q = len(kernelX[0])
        Gx = np.zeros((dimX,dimY),int)
        Gy = np.zeros((dimX,dimY),int)
        
        for i in range(dimX-P):
                for j in range(dimY-Q):
                        Gx[i,j] = (kernelX * inImage[i:i+P,j:j+Q]).sum()
                        Gy[i,j] = (kernelY * inImage[i:i+P,j:j+Q]).sum()
        return Gx,Gy

def gradientImage(inImage,operator):
        if (operator == 'Roberts'):
                Gx,Gy = _Roberts(inImage)
        elif(operator == 'CentralDiff'):
                Gx,Gy = _CentralDiff(inImage)
        elif(operator == 'Prewitt'):
                Gx,Gy = _Prewitt(inImage)
        elif(operator == 'Sobel'):
                Gx,Gy = _Sobel(inImage)
        return np.abs(Gx),np.abs(Gy)
def direction(dk):
        if(dk <120):
                return (-1,0)
        elif(dk>150 and dx< 90):
                pass

        
                        
        
def Canny(inImage, sigma, tlow,thigh):
        dimX = len(inImage)
        dimY = len(inImage[0])

        smoothImage = gaussianFilter(inImage,sigma)
        Gx,Gy = gradientImage(inImage,'Sobel')
        
        Em = np.sqrt(Gx**2+Gy**2)
        E0 = np.arctan2(Gy,Gx)
        In = np.zeros((dimX,dimY))
        E0 = np.rad2deg(E0)
        
        for i in range(1,dimX-1):
                for j in range(1,dimY-1):
                        if(E0[i,j]<E0[i-1,j-1] and E0[i,j] <E0[i+1,j+1]):
                                In[i,j] = 0
                        else:
                                In[i,j] = E0[i,j]
        


#Imagenes de prueba para operadores binarios
def createLines():
        return np.array([[255,0,0,0],\
                   [255,0,0,0],\
                   [0,255,255,0],\
                   [0,255,0,0],\
                   [0,255,0,0],\
                   [0,255,0,0]])
def testHitOrMiss():
        matriz = np.zeros([7,7])
        matriz[1:2, 3:5] = 1
        matriz[2:3, 2:6] = 1
        matriz[3:4, 2:6] = 1
        matriz[4:5, 3:5] = 1
        matriz[5:6, 3:4] = 1
        return matriz

if __name__ == '__main__':
        
        inImage = cv2.imread(PATH+'grays.png',0)
        #outImage = erode2(inImage,np.ones((5,5)))
        #showImage(inImage,outImage)
        #print(inImage)
        #outImage=equalizeIntensity(inImage,256)
        #print(outImage)
        #showHistogram(inImage,outImage)
        
        outImage = adjustIntensity(inImage,[0.47,0.56])
        #outImage2 = adjustIntensity(inImage,[0,1])
        #print(outImage)
        showHistogram(inImage,outImage,[256,1])
        