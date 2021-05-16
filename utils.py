import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def showImage(img,size=10):
    '''
    Displays a single image, while also converting from BGR to RGB
    '''
    plt.figure(figsize=(size,size))
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
def showImageGray(img,size=10):
    '''
    Displays grayscale images
    '''
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap="gray")
    
    

def trimBordersX(img,tolerance = 30):
    '''
    Trims left and right borders of an image.
    Returns the amount of trimming to be done on either side. 
    '''
    
    gFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        

    # Determing starting x
    startX = 0
    for x in range(0,gFrame.shape[1]):
        max =gFrame[:,x].max()
        min =gFrame[:,x].min()
        # print(avg)
        if max - min > tolerance:
            break
        startX = x + 1

    # Determine ending x
    endX = gFrame.shape[1]-1
    for x in range(gFrame.shape[1]-1,0,-1):
        max =gFrame[:,x].max()
        min =gFrame[:,x].min()
        if max - min > tolerance:
            break
        endX = x - 1
    
    return startX, gFrame.shape[1] - endX


def trimBordersY(img,tolerance = 30):
    '''
    Trims top and bottom borders of an image.
    Returns the amount of trimming to be done on either side. 
    '''
     
    gFrame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    
    # Determing starting x
    startY = 0
    for y in range(0,gFrame.shape[0]):
        max =gFrame[y,:].max()
        min =gFrame[y,:].min()
        if max - min > tolerance:
            break
        startY = y + 1

    # Determine ending x
    endY = gFrame.shape[0]-1
    for y in range(gFrame.shape[0]-1,0,-1):
        max =gFrame[y,:].max()
        min =gFrame[y,:].min()
        if max - min > tolerance:
            break
        endY = y - 1
    
    return startY, gFrame.shape[0] - endY

def trimImage(im,tolerance=30):
    '''
    Return a trimmer images
    '''    
    diffSX, diffEX = trimBordersX(im,tolerance)
    diffSY, diffEY = trimBordersY(im,tolerance)
    
    return im[diffSY:-diffEY,diffSX:-diffEX]

def adjustContrast(img,c):
    '''
    Adjust contrast of an image
    c: the multiplying factor
    '''
    return np.clip(img * c, 0,255)


def padBox(frame, startX,startY,endX,endY,padX=0,padY=0):
    '''
    padX, padY: No of pixels to pad on X and Y respectively
    '''
    return max(0,startX-padX), \
            max(0,startY-padY), \
            min(frame.shape[1]-1, endX+padX), \
            min(frame.shape[0]-1, endY+padY)



def resize(img,max_width=2400):
    '''
    Resizes image to a specific width
    '''
    height, width  = img.shape[:2]
    if width > max_width: 
        perc = max_width/width
        newHeight = int(height * perc)
        img = cv2.resize(img, (max_width,newHeight))                
                         
    return img
    
                         
def improveContrast(img):  
    # Move to HSV colour space
    im_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Get H, S, V channels separately
    H, S, V = cv2.split(im_hsv)

    # create a CLAHE object.
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10,10))
    V_eq = clahe.apply(V)

    V_eq = (adjustContrast(V_eq,1.2)).astype(np.uint8)

    # Merge H and S channels with the equalised V channel
    im_eq= cv2.merge((H,S,V_eq))
    im_eq = cv2.cvtColor(im_eq,cv2.COLOR_HSV2BGR)

    return im_eq

def imgPreprocess(img):
    img = resize(img)
    return improveContrast(img)
    
    
    