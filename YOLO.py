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


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("C:\\Users\\me\\Documents\\Python\\YOLO\\yolov3-tiny.cfg", "C:\\Users\\me\\Documents\\Python\\YOLO\\yolov3-tiny.weights")


image = cv2.imread("image.jpeg")
image_size = 256
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size=(256, 256), swapRB=True)


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 256),
	swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

boxes = []
confidences = []
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
