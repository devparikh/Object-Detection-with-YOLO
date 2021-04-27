# We are importing all of the dependencies that we need in order to build this project 
import cv2
import numpy as np
import random

yolo_network = cv2.dnn.readNetFromDarknet("/content/yolov3-tiny.cfg","/content/yolov3-tiny.weights")

classes = []
# we are opening the file using the open statement
with open("/content/coco.names", "r") as r:
  # here we are reading the image in and doing splitline which splits a string into a list
  classes = r.read().splitlines()
# this should be 80
len(classes)

image = cv2.imread("image.jpeg")
image_size = 256
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size=(256, 256), swapRB=True)

# getting the input as this new_image
yolo_network.setInput(blob)
# returns the indexes of layers with unconnected outputs
output_layer = yolo_network.getUnconnectedOutLayersNames()

layersoutput = yolo_network.forward(output_layer)

bounding_boxes = []
confidence = []
classid = []

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
      thres = 0.6
      # we will now create the 
      if confidence_score > thres:
        bounding_box = detection[0:4] * np.asarray([image_size,image_size,image_size,image_size])
        # the astype method converts any value in any form to whatever form specified
        (x_center, y_center, image_size, image_size) = bounding_box.astype("int")

        x = int(x_center - (0.5*image_size))
        y = int(y_center - (0.5*image_size))

        classid.append(int(object_class))
        confidence.append(int(confidence_score))
        bounding_boxes.append(bounding_box)
        
# non-max-supression
nms_threshold = 0.4
nms = cv2.dnn.NMSBoxes(bounding_boxes, confidence, thres, nms_threshold)

COLOURS = []
while len(COLOURS) < 80:
  red = random.randint(0, 255)
  green = random.randint(0, 255)
  blue = random.randint(0, 255)
  colour = [red, green, blue]
  COLOURS.append(colour)
  
  
LABEL = []
def draw_bounding_box(confidence, bounding_boxes, classid, classes, COLOURS):
  if len(nms) > 0:
    for i in nms.flatten():
      for i in classid:
        label = classes[i]
        LABEL.append(label)
        
    colours = random.choice(COLOURS)
    cv2.rectangle((x,y), (x+image_size, y+image_size), colours, 2)
    cv2.putText(blob, "class" + classid + "confidence" + confidence, (x,y), cv2.FONT_HERSHEY_COMPLEX, colours, 2)
    
draw_bounding_box(confidence, bounding_boxes, classid, COLOURS, classes)