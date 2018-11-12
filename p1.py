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
        arrImage = inImage
        maxValue = np.amax(arrImage)
        minValue = np.amin(arrImage)
        print('max: ',maxValue,'min ',minValue)
        
        if (inRange == []):
                inRange = np.array([minValue,maxValue],dtype=int)
                
        
        dimX = len(arrImage)
        dimY = len(arrImage[0]) 
        outImage = np.zeros((dimX,dimY),"f")
        
        for i in range(dimX):
                for j in range(dimY):
                        outImage[i,j] = inRange[0] + ((inRange[1]-inRange[0])*((arrImage[i,j]-minValue)) / (maxValue-minValue))
        
        
        
        return outImage/255


        
def equalizeIntensity(inImage, nbins = 256):
        #inImage = cv2.imread(PATH+inImage,0)
        arrImage = inImage
        dimX = len(arrImage)
        dimY = len(arrImage[0]) 
        outImage = np.zeros((dimX,dimY),"f")
        
        histAcum,bins,x =pyplot.hist(arrImage.ravel(),bins=256,cumulative=True)
        hist,bins,x =pyplot.hist(arrImage.ravel(),bins=256)

        cdf = (histAcum/(dimX*dimX)) *255
        for i in range(dimX):
                for j in range(dimY):
                        outImage [i,j]= cdf[arrImage[i,j]] 
        
        return outImage.astype(int)
        
        
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
        return np.abs(Gx),np.abs(Gy)

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
def Canny(inImage, sigma, tlow,thigh):
        dimX = len(inImage)
        dimY = len(inImage[0])

        smoothImage = gaussianFilter(inImage,sigma)
        Gx,Gy = gradientImage(inImage,'Sobel')
        Em = np.sqrt(Gx**2+Gy**2)
        E0 = np.arctan(Gy/Gx)
        print(E0)
        



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
        
        inImage = cv2.imread(PATH+'lena.png',0)
        """
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
        
        #kernel = np.array([[0.1,0.2,0.1]])
        
        
        outImage = filterImage(inImage,kernel)
        #showHistogram(inImage,outImage)
        print(outImage)
        """
        #outImage = medianFilter(inImage,3)
       # outImage = highBoost(inImage,1.5,'gaussian',3.6)
        #cv2.imwrite('outImage.png',outImage)
        #print(inImage)
        #outImage = adjustIntensity(outImage,[0,255])
        #showHistogram(inImage,outImage)
        #print(gaussKernel1D(1))
        #print(gaussKernel1D(1.2))
        
        
        #Operadores morfologicos
        #inImage = cv2.imread(PATH+'example.png',0)
        
        #inImage = createLines()
        #kernelM = np.array([[1,1]])
        #outImage = erode(inImage,kernelM)
        #outImage = dilatation(inImage,kernelM,[0,1])
        #testImage = cv2.erode(inImage,kernelM.astype(np.uint8),8)
        #print(outImage)
        #showHistogram(inImage,outImage)
        
        Canny(inImage,3,0,0)
        """
        kernel = np.array([[0,0,0],\
                           [1,1,0],\
                           [0,1,0]])
        
        kernelB = np.array([[0,1,1],\
                            [0,0,1],\
                            [0,0,0]])
        
        inImage = testHitOrMiss()
        #inImage = cv2.imread(PATH+'binary1.png',0)
        outImage1 = erode(_toNormal(inImage),kernel,[1,1])
        outImage2 = erode(_toNormal(complementario(inImage)),kernelB,[1,1])
        outImage = hit_or_miss(inImage,kernel,kernelB,[1,1])
        outImage = outImage1 & outImage2
        showHistogram(inImage,outImage2)
        
        #outImage = adjustIntensity(inImage,[100,200])
        #Gx,Gy = gradientImage(inImage,'Prewitt')
        #outImage = np.abs(Gx+Gy)
        #showImage(inImage,outImage)
        """
        