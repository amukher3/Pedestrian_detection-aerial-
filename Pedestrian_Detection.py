# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:09:58 2020

@author: Abhishek Mukherjee
"""

import numpy as np
import argparse
import cv2

## Parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#ap.add_argument("-p", "--prototxt", required=True,
#	help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
#	help="path to Caffe pre-trained model")
#ap.add_argument("-c", "--confidence", type=float, default=0.2,
#	help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())
#
## List of class labels MobileNet SSD was trained to detect, then generate 
## a set of bounding box colors for each class
#
##CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
##	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
##	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
##	"sofa", "train", "tvmonitor","sun","orange"]
#
#
#CLASSES = ["person"]
#
#COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
#
#print("Loading model...")
#
#for i in range(10,10,100):
#    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#    image = cv2.imread(args["image"])
#    (h, w) = image.shape[:2]
#    #blob = cv2.dnn.blobFromImage(cv2.resize(image, (1600, 1600)), 0.007, (1600,1600), 127.5)
#    blob = cv2.dnn.blobFromImage(cv2.resize(image, (1600, 1600)), 0.007, (1600,1600),i)
#    print("Computing object detections...")
#    net.setInput(blob)
#    detections = net.forward()
#
#    # looping over the detections
#    for i in np.arange(0, detections.shape[2]):
#
#        confidence = detections[0, 0, i, 2]
#
#        if confidence > args["confidence"]:
#
#            idx=int(detections[0, 0, i, 1])
#            box=detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#            (startX, startY, endX, endY) = box.astype("int")
#            label = "{}: {:.2f}%".format(CLASSES[0], confidence * 100)
#            ("[INFO] {}".format(label))
#            cv2.rectangle(image, (startX, startY), (endX, endY),
#			COLORS[0], 2)
#            y = startY - 15 if startY - 15 > 15 else startY + 15
#		#cv2.putText(image, label, (startX, y),
#		#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
#
#    # Output image
#    cv2.imshow("Output", image)
#    #cv2.waitKey(0)

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# List of class labels MobileNet SSD was trained to detect, then generate 
# a set of bounding box colors for each class

#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#	"sofa", "train", "tvmonitor","sun","orange"]


CLASSES = ["person"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
#blob = cv2.dnn.blobFromImage(cv2.resize(image, (1600, 1600)), 0.007, (1600,1600), 127.5)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (1600, 1600)), 0.007, (1600,1600),5)
print("Computing object detections...")
net.setInput(blob)
detections = net.forward()

# looping over the detections
for i in np.arange(0, detections.shape[2]):

	confidence = detections[0, 0, i, 2]

	if confidence > args["confidence"]:

		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		label = "{}: {:.2f}%".format(CLASSES[0], confidence * 100)
		#print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[0], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		#cv2.putText(image, label, (startX, y),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

#Output image
cv2.imshow("Output", image)
#print(img.empty())
#print(img)
cv2.waitKey(0)
#cv2.imwrite("C:/Users/abhi0/OneDrive/Documents/object_detection/images/output_25.jpeg",image)