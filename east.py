import cv2
import numpy as np
import math

import os
import requests
import tarfile


def downloadAndUntarFile(url,filename):
    # Download file
    response = requests.get(url, stream = True)
    file = open(filename,"wb")
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)
    file.close()
    
    # Untar file
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()

def detectText(img,score_threshold = 0.1, nms_threshold=0.1):
    '''
    This runs the EAST text detection algorithm
    From: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
    '''
    height, width = img.shape[:2]
    
    # Download the pre-trained EAST model if not already present
    # From: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
    modelURL = "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
    modelFilename = "frozen_east_text_detection.pb"
    
    if not os.path.isfile(modelFilename):
        downloadAndUntarFile(modelURL, "frozen_east_text_detection.tar.gz")
        
    # Read the model
    net = cv2.dnn.readNet(modelFilename)
    
    # Resize image to multiples of 32 (required by the EAST model)
    imH,imW = int(height/32)*32, int(width/32)*32
    frame = cv2.resize(img,(imW,imH))
    
    # Get a blob of the image, which will be an input to the network
    blob = cv2.dnn.blobFromImage(frame, 
                            1.0,                        # pixel scaling
                            (imW, imH),                 # input dimensions
                            (123.68, 116.78, 103.94),   # the mean that needs to be subtracted
                            True,                       # whether to convert BGR to RGB
                            False)                      # whether to crop the image
    
    
    
    # Define output layers of the network
    outputLayers = ["feature_fusion/Conv_7/Sigmoid",
                    "feature_fusion/concat_3"]
    
    # Pass the image through the network
    # This is where the prediction happens
    net.setInput(blob)
    output = net.forward(outputLayers)
    
    scores = output[0]
    geometry = output[1]
    
    # Decode the output of the network
    [boxes, confidences] = decode(scores, geometry,0.1)
    
    # Create a new frame on which boxes will be drawn
    framev = np.copy(frame)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences,\
                                      score_threshold = score_threshold,\
                                      nms_threshold=nms_threshold)

    # Draw the boxes on the image
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(framev, p1, p2, (0, 255, 0), 4)

    # Return the 
    #  resized frame, 
    #  frame with boxes
    #  bounding boxes
    #  indices of selected bounding boxes
    return frame, framev, boxes, indices 




def decode(scores, geometry, scoreThresh):
    '''
    Used for decoding the output of the deep network.
    From: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
    '''
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                       offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w, h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]
