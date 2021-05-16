import cv2
import utils
from matplotlib import pyplot as plt

def getBBox(rotatedBox,padX=0,padY=0):
    '''
    Get the bounding box from the rotated box
    '''
    v = cv2.boxPoints(rotatedBox)
    startX,endX = int(v[:,0].min()),int(v[:,0].max())
    startY,endY = int(v[:,1].min()),int(v[:,1].max())
    return max(startX-padX,0),endX+padX,max(startY-padY,0),endY+padY

def getSortedBoxes(boxes,indices,npad=0):
    '''
    Returns a set of reduced, selected and sorted bounding boxes
    boxes: the set of bounding boxes
    npad: negative padding in fraction (0-1)
    '''
    # Create a list of selected boxes along with min(x) and min(y)
    sboxes =[]

    for i in indices:
        rtBox = boxes[i[0]]
        
        startX,endX,startY,endY = getBBox(rtBox)
        
        reduc = -int(((endX-startX)/2) * npad)
        
        startX,endX,startY,endY = getBBox(rtBox,padX=reduc)

        if endX - startX <=0:
            continue

        sboxes.append({ "box":rtBox,
                        "minX": startX  ,
                        "minY": startY,
                        "maxX": endX,
                        "maxY": endY,
                        "centX": int((startX + endX)/2),
                        "centY": int((startY + endY)/2),
                        })

    # Sort sboxes based on minX and minY
    sboxes.sort(key= lambda e: e["minX"]*100 + e["minY"])
    return sboxes


def trimBoxes(frame, boxes):
    '''
    Eliminates unnecessary borders from bounding boxes
    '''
    tboxes = []
    for b in boxes:
        diffStartX, diffEndX = utils.trimBordersX(frame[ b["minY"]:b["maxY"], b["minX"]:b["maxX"]])
        b["minX"] += diffStartX
        b["maxX"] -= diffEndX
        if b["maxX"] - b["minX"] <= 0:
            continue
        tboxes.append(b)

    return tboxes


def newBook(sbox, minX = 0):
    '''
    Returns a new book dict
    '''
    return {"minX":minX,
            "minY":sbox["minY"],
            "maxX":sbox["maxX"],
            "maxY":sbox["maxY"],
            "boxes":[sbox]}

def addBoxToBook(book,sbox):
    '''
    Appends a new box to the book dict
    '''
    book["boxes"].append(sbox)
    book["minX"] = min(book["minX"],sbox["minX"])
    book["minY"] = min(book["minY"],sbox["minY"])
    book["maxX"] = max(book["maxX"],sbox["maxX"])
    book["maxY"] = max(book["maxY"],sbox["maxY"])
    return book


def boxesToBooks(boxes, xMargin=5, debug=False):
    '''
    Segregates boxes into separate books
    xMargin: Margin by which books are separated in the x (column) direction
    '''
    # Initialise the book list
    bookList = []
    # Initialise the first book
    book = newBook(boxes[0])
    for b in boxes[1:]:
        newMinX =  b["minX"]

        # Start a new book if newMinX is greater than book["maxX"] by some margin
        if  newMinX-book["maxX"]>xMargin:
            if debug:
                print("current minX, maxX for book:", newMinX, book["maxX"])
                print("New Book")

            # Calculate the midpoint of two books
            midPoint = int((newMinX + book["maxX"])/2)
            book["maxX"] = midPoint-1

            # Append existing book to book list
            bookList.append(book)

            # Start a new book
            book = newBook(b,midPoint+1)
            continue

        book = addBoxToBook(book,b)

    # Append the last book
    bookList.append(book)
    return bookList

def getBookImages(frame,bookList,padX=10,padY=500):
    '''
    Get book images using the bounding box of the book
    '''
    bookImages = []
    for bk in bookList:
        # Add padding to make sure the entire book is included
        startX,startY,endX,endY = utils.padBox(frame, bk["minX"],bk["minY"],\
                                               bk["maxX"], bk["maxY"],\
                                               padX=padX,padY=padY)
        block = frame[startY:endY, startX:endX]
        
        # Trim the top and bottom parts that are not important
        diffStartY, diffEndY = utils.trimBordersY(block,100)
        
        bookImages.append(block[diffStartY:-diffEndY])
        
    return bookImages


def showSegmentedBooks(bookImages,path=None):
    '''
    Displays all segmented books
    '''
    fig, ax = plt.subplots(nrows=1,ncols=len(bookImages),figsize=(20,10))

    for i, bi in enumerate(bookImages):
        ax[i].imshow(cv2.cvtColor(bi,cv2.COLOR_BGR2RGB))
        ax[i].set_title("Book - {} ".format(i+1))
        ax[i].axis('off')
        
    if path is not None:
        fig.savefig(path)