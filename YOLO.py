# We are importing all of the dependencies that we need in order to build this project 
import cv2
import numpy as np
import random

yolo_network = cv2.dnn.readNetFromDarknet("C:\\Users\\me\\Documents\\Python\\YOLO\\yolov3-tiny.cfg","C:\\Users\\me\\Documents\\Python\\YOLO\\yolov3-tiny.weights")

classes = []
# we are opening the file using the open statement
with open("C:\\Users\\me\\Documents\\Python\\YOLO\\coco.names", "r") as r:
  # here we are reading the image in and doing splitline which splits a string into a list
  classes = r.read().splitlines()
# this should be 80
print(len(classes))

image = cv2.imread("image.jpeg")
image_size = 256
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size=(256, 256), swapRB=True)



# returns the indexes of layers with unconnected outputs
ln = yolo_network.getLayerNames()
output_layer = [ln[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()]


image = cv2.imread("image.jpeg")
image_size = 256
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size=(256, 256), swapRB=True)

# getting the input as this new_image
yolo_network.setInput(blob)

layersoutput = yolo_network.forward(output_layer)



boxes = []
confidences = []
classid = []  
thres = 0.5

# now we need to retrieve all of the data from the model
for output in layersoutput:
   for detection in output:
      # every thing after the 4th index in the vector returned by the model can be used for the scores
      # because the previous 4 indexes are height, width and the x and y coordinates of the center point of each grid in the image
      # after the specifications of the bounding box the next thing is to get the class and the confidence score 
      # everthing after the 4th index will be a class as there are 80 classes in the dataset
      scores = detection[5:]
      # we want to find the highest value in the distribution probablity of the classes and that will be our class
      object_class = np.argmax(scores)
      # now you want wha the confidence score is of the object in the grid
      # the best thing to do is to find the probablity that the object class has and save that as the confidence
      confidence_score = scores[object_class]
      # this is the threshold that we have put to filter out any weak detections
      
      # we will now create the 
      if confidence_score > thres:
        bounding_box = detection[0:4] * np.asarray([image_size,image_size,image_size,image_size])
        # the astype method converts any value in any form to whatever form specified
        (x_center, y_center, width, height) = bounding_box.astype("int")

        x = int(x_center - (width / 2))
        y = int(y_center - (height / 2))

        boxes.append([x, y, int(image_size), int(image_size), bounding_box])
        confidences.append(float(confidence_score))
        classid.append(object_class)  

# non-max-supression
nms_threshold = 0.4
nms = cv2.dnn.NMSBoxes(boxes, confidences, thres, nms_threshold)

COLOURS = []
while len(COLOURS) < 80:
  # here we essentially just want to find a random red value, green value, and blue value to come up with a rgb colour
  red = random.randint(0, 255)
  green = random.randint(0, 255)
  blue = random.randint(0, 255)
  # we are going to create an rgb colour here
  colour = [red, green, blue]
  # appending this colour to the list to it can be used later
  COLOURS.append(colour)

# here we are checking if there is any bounding boxes after non-maxima supression
if len(nms) > 0:
  # we collapse the array into a 1-dimensional vector to iterate over
  for i in nms.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])
      # we want to choice a random colour from the COLOURS list
      colours = random.choice(COLOURS)
      # drawing the bounding box
      cv2.rectangle(image, (x,y), (x+width, y+height), colours, 2)
      text  = "{}: {}".format(classes[classid[i]], confidences[i])
      # labelling the bounding box
      cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, colours, 2)
# displaying the bounding box
cv2.imshow("Image", image)
cv2.waitKey(40)

cv2.imwrite("image2.jpeg", image)
classIDs = []
thres = 0.5

for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > thres:
			box = detection[0:4] * np.array([image_size, image_size, image_size, image_size])
			(centerX, centerY, width, height) = box.astype("int")

			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
print(scores)
# non-max-supression
nms_threshold = 0.7
nms = cv2.dnn.NMSBoxes(boxes, confidences, thres, nms_threshold)

COLOURS = []
while len(COLOURS) < 80:
  # here we essentially just want to find a random red value, green value, and blue value to come up with a rgb colour
  red = random.randint(0, 255)
  green = random.randint(0, 255)
  blue = random.randint(0, 255)
  # we are going to create an rgb colour here
  colour = [red, green, blue]
  # appending this colour to the list to it can be used later
  COLOURS.append(colour)

if len(nms) > 0:
	for i in nms.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		color = [int(c) for c in COLOURS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

cv2.imshow("Image", image)
cv2.imwrite("image.jpeg", image)
cv2.waitKey(0)
