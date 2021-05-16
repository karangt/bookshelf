# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

import utils


# %%
cv2.__version__


# %%
# Read the model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")


# %%
# Read the image
img = cv2.imread("./bookshelf.jpg")
height, width = img.shape[:2]

print(height,width)


# %%
# Resize image to multiples of 32 (required by the EAST model)
imH,imW = int(height/32)*32, int(width/32)*32
frame = cv2.resize(img,(imW,imH))
frame.shape


# %%
blob = cv2.dnn.blobFromImage(frame,
                            1.0,                        # pixel scaling
                            (imW, imH),            # input dimensions
                            (123.68, 116.78, 103.94),   # the mean that needs to be subtracted
                            True,                       # whether to convert BGR to RGB
                            False)                      # whether to crop the image


# %%
# Define output layers of the network
outputLayers = ["feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]


# %%
# Pass the image through the network
net.setInput(blob)
output = net.forward(outputLayers)


# %%
scores = output[0]
geometry = output[1]

print ("Scores:",scores.shape)
print ("Geometry:",geometry.shape)


# %%
[boxes, confidences] = utils.decode(scores, geometry,0.1)


# %%
# apply non-maxima suppression to suppress weak, overlapping bounding boxes
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, score_threshold = 0.1, nms_threshold=0.1)

for i in indices:
	# get 4 corners of the rotated rect
	vertices = cv2.boxPoints(boxes[i[0]])
	for j in range(4):
		p1 = (vertices[j][0], vertices[j][1])
		p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
		cv2.line(frame, p1, p2, (0, 255, 0), 1)

# show the output image
plt.figure(figsize=(20,20))
plt.imshow(frame)


# %%
help(cv2.dnn.NMSBoxesRotated)


# %%


